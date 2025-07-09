import os
import json
import asyncio
import sys
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import AzureOpenAI

from crawl4ai import AsyncWebCrawler

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()

# Configure Azure OpenAI client
def get_openai_client():
    try:
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",  # Updated to stable version
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/')  # Ensure no trailing slash
        )
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        return None

def clean_and_validate_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return None
        if result.scheme not in ('http', 'https'):
            return None
        return url
    except ValueError:
        return None

def google_search(query: str, num_results=5) -> list[str]:
    print(f"🔍 Searching Google for: {query}")
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("Missing SERPER_API_KEY in .env")
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        return [clean_and_validate_url(r["link"]) for r in data.get("organic", []) if "link" in r]
    except Exception as e:
        print(f"❌ Google search failed: {e}")
        return []

async def crawl_urls(urls: list[str]) -> list:
    if not urls:
        return []
    print(f"🌐 Crawling {len(urls)} URLs...")
    async def crawl_single(url):
        try:
            async with AsyncWebCrawler() as crawler:
                # Modified to get both raw and cleaned content
                result = await crawler.arun(
                    url=url,
                    by="text",
                    extract_rules={"main_content": "body"},  # Extract all body content
                    include_links=False,
                    include_images=False,
                    include_tables=False,
                    bypass_cache=True,
                    return_raw_html=True  # Ensure we get raw HTML
                )
                return result
        except Exception as e:
            print(f"⚠️ Failed to crawl {url}: {e}")
            return None
    return await asyncio.gather(*[crawl_single(url) for url in urls])

def extract_content(page) -> tuple[str, str]:
    """Extract both text and HTML content from page"""
    if not page:
        return "", ""
    
    # Try to get cleaned text first
    text = getattr(page, "cleaned_text", "") or getattr(page, "text", "")
    
    # Fallback to raw HTML extraction
    if not text:
        html = getattr(page, "raw_html", "") or getattr(page, "cleaned_html", "")
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
            text = soup.get_text(separator='\n', strip=True)
    
    # Get URL
    url = getattr(page, "url", "Unknown source")
    
    return text[:10000], url  # Limit to 10k chars

def build_openai_prompt(query: str, text_chunks: list[str], sources: list[str]) -> list[dict]:
    content_text = "\n\n".join(text_chunks)
    refs = "\n".join(f"- {url}" for url in sources if url)

    return [
        {
            "role": "system",
            "content": "You are a skilled research assistant. Generate clear, fact-based content using only the provided sources."
        },
        {
            "role": "user",
            "content": f"""Create a concise article about: {query}

Sources:
{content_text[:10000]}  # Strict limit to avoid token limits

References:
{refs}

Requirements:
- 300-500 words
- Neutral, informative tone
- Use only provided content
- Markdown format with headings
- Include reference section
"""
        }
    ]

async def summarize_with_openai(client, prompt_messages):
    if not client:
        return None
    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),  # Verify this matches your Azure deployment
                messages=prompt_messages,
                temperature=0.3,
                max_tokens=1500,
            )
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API error: {str(e)}")
        return None

async def main():
    user_query = input("📝 Enter your topic or question: ").strip()
    if not user_query:
        print("❌ You must enter a query.")
        return

    client = get_openai_client()
    if not client:
        return

    # Step 1: Search
    urls = [url for url in google_search(user_query) if url][:5]
    print(f"🔗 Found {len(urls)} URLs")

    # Step 2: Crawl
    pages = await crawl_urls(urls)

    # Step 3: Process content - improved extraction
    text_chunks = []
    sources = []
    for page in pages:
        if page:
            text, url = extract_content(page)
            if text:
                print(f"✅ Extracted {len(text)} chars from {url[:50]}...")
                text_chunks.append(text)
                sources.append(url)

    if not text_chunks:
        print("❌ No content extracted. Possible reasons:")
        print("- Pages require JavaScript rendering (try browser_mode=True)")
        print("- Paywalled or restricted content")
        print("- Extraction rules too strict")
        return


    # Step 4: Generate summary
    prompt = build_openai_prompt(user_query, text_chunks, sources)
    print("🧠 Generating article...")
    article_md = await summarize_with_openai(client, prompt)

    if not article_md:
        print("❌ Failed to generate article.")
        return

    # Step 5: Save
    fname = f"summary_{user_query.replace(' ', '_')[:30]}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(article_md)
    print(f"\n✅ Article saved to '{fname}'")
    print("\n=== Generated Summary ===")
    print(article_md[:1000] + "...")  # Print first part of summary

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")