from dotenv import load_dotenv
load_dotenv()  # Load .env into os.environ

import os
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import LLMConfig
from crawl4ai.content_filter_strategy import LLMContentFilter

async def test_llm_filter():
    #url = "https://thehackernews.com/"
    #url = "https://timesofindia.indiatimes.com/"
    url = "https://www.theage.com.au/"
    
    
    
    # 1. Browser configuration
    browser_config = BrowserConfig(headless=True, verbose=True)
    run_config     = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
    
    # 2. Crawl HTML
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url, config=run_config)
        html   = result.cleaned_html

        # 3. Azure OpenAI configuration with correct param names
        azure_llm = LLMConfig(
            provider=f"azure/{os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')}",
            api_token=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # 4. Set up content filter
        content_filter = LLMContentFilter(
            llm_config=azure_llm,
            instruction="""
Extract the news articles from the homepage. For each article, provide:

You are a content summarization AI.

Given a news article's HTML or cleaned text, extract and generate a JSON object with:

- "title": exact headline
- "summary": rephrase the article into a ~500-word summary
- "Snippet": two line summary for the article
- "full_story": rewrite the full article in ~1000 words, preserving facts and structure
- "image_url": the main image if any (from the article or OG tags)
- "reference": create the url to provide refrence to the article

Return a valid JSON only. Do not wrap it in Markdown or extra text.
""",
            chunk_token_threshold=1024,
            ignore_cache=True,
            verbose=True
        )
        
        # 5. Filter and fetch results
        filtered_content = await asyncio.to_thread(content_filter.filter_content, html)
        
        # 6. Display and save
        print(f"\nFiltered Content Length: {len(filtered_content)} lines\n")
        for line in filtered_content[:20]:
            print(line)
        
        with open("filtered_content.md", "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_content))
        
        # 7. Show token usage
        content_filter.show_usage()

if __name__ == "__main__":
    asyncio.run(test_llm_filter())
