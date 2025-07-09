import requests
from newspaper import Article
from transformers import pipeline
import time

# === CONFIGURATION ===
SERPAPI_KEY = "07bbea4e794fceda9fc4a7571d2e56661ebb7c6b"  # Replace with your actual SerpAPI key
MODEL_NAME = "tiiuae/falcon-rw-1b"  # Smaller model for easier local use
MAX_CHUNKS = 3  # Limit to avoid overloading the LLM
MAX_WORDS_PER_CHUNK = 200

# === STEP 1: DISCOVER ===
def discover_with_serpapi(query, limit=5):
    params = {
        "engine": "google",
        "q": query,
        "num": limit,
        "api_key": SERPAPI_KEY
    }
    resp = requests.get("https://serpapi.com/search.json", params=params)
    data = resp.json()
    hits = []
    for i, item in enumerate(data.get("organic_results", []), start=1):
        hits.append({
            "id": i,
            "url": item.get("link"),
            "title": item.get("title"),
            "snippet": item.get("snippet")
        })
    return hits

# === STEP 2: EXTRACT TEXT FROM URL ===
def extract_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return ""

# === STEP 3: CHUNK TEXT ===
def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# === STEP 4: SUMMARIZE WITH CITATIONS ===
def summarize_with_hf(user_query, sources):
    chunks_with_refs = []
    for source in sources:
        for chunk in chunk_text(source["text"], max_words=MAX_WORDS_PER_CHUNK):
            chunks_with_refs.append((source["id"], chunk))
            if len(chunks_with_refs) >= MAX_CHUNKS:
                break
        if len(chunks_with_refs) >= MAX_CHUNKS:
            break

    context = "\n".join([f"[{cid}] {chunk}" for cid, chunk in chunks_with_refs])
    prompt = f"""
You are an assistant that summarizes research from multiple sources.

Given the following context:
{context}

Answer the following question using citations (e.g., (1), (2)) where appropriate:

Question: {user_query}
"""

    summarizer = pipeline("text-generation", model=MODEL_NAME, max_new_tokens=300)
    result = summarizer(prompt, do_sample=False)[0]["generated_text"]
    return result

# === MAIN DRIVER ===
def run_perplexity_clone(query):
    print(f"Searching for: {query}")
    hits = discover_with_serpapi(query)
    print(f"Found {len(hits)} links. Extracting content...")

    sources = []
    for hit in hits[:MAX_CHUNKS]:
        print(f"- {hit['title']}")
        text = extract_text(hit['url'])
        if len(text) > 500:
            sources.append({**hit, "text": text})
        time.sleep(1.5)  # polite crawling

    if not sources:
        print("No content could be extracted.")
        return

    print("\nGenerating answer using Hugging Face model...")
    answer = summarize_with_hf(query, sources)

    print("\n=== FINAL ANSWER ===")
    print(answer)
    print("\n=== SOURCES ===")
    for src in sources:
        print(f"({src['id']}) {src['title']} - {src['url']}")

# Example usage
if __name__ == "__main__":
    run_perplexity_clone("What is Perplexity AI?")
