import requests
from newspaper import Article
from transformers import pipeline
import time
import logging
from bs4 import BeautifulSoup
import json
import torch
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
SERPER_API_KEY = 'fb4c4d86127f32f8f6fd5b7cf3931e9c51997aa2'
MODEL_NAME = "facebook/bart-large-cnn"
MAX_ARTICLES = 10
SUMMARY_LENGTH = 400
REQUEST_DELAY = 1.0
MAX_RETRIES = 2

# Browser-like headers to prevent blocking
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/'
}

# Initialize device once
DEVICE = 0 if torch.cuda.is_available() else -1
logger.info(f"⚙️ Running on {'GPU' if DEVICE == 0 else 'CPU'}")

def verify_serper_api():
    """Verify API connectivity with comprehensive error handling"""
    test_payload = {
        'q': "test",
        'num': 1,
        'gl': 'us',
        'hl': 'en'
    }
    try:
        resp = requests.post(
            'https://google.serper.dev/news',
            headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
            json=test_payload,
            timeout=10
        )
        
        if resp.status_code == 401:
            logger.error("❌ Serper.dev API: Unauthorized (Invalid API Key)")
            logger.info("ℹ️ Get your API key from: https://serper.dev/dashboard")
            return False
        elif resp.status_code == 403:
            logger.error("❌ Serper.dev API: Forbidden (Check Usage Limits)")
            return False
        resp.raise_for_status()
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API Connection Failed: {str(e)}")
        return False

def extract_article_content(url):
    """Robust content extraction with multiple fallback methods"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Method 1: Newspaper3k with custom headers
            article = Article(url, headers=HEADERS, request_timeout=15)
            article.download()
            article.parse()
            
            # Method 2: Direct HTML parsing if newspaper fails
            if not article.text or len(article.text) < 100:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                if resp.status_code == 403:
                    raise Exception(f"403 Forbidden: {url}")
                soup = BeautifulSoup(resp.text, 'html.parser')
                article.text = ' '.join([p.get_text() for p in soup.find_all('p')])
            
            # Get best available image
            top_image = article.top_image
            if not top_image:
                meta_image = soup.find('meta', property='og:image')
                top_image = meta_image['content'] if meta_image else None
            
            return {
                "full_text": article.text,
                "authors": article.authors,
                "publish_date": str(article.publish_date),
                "top_image": top_image
            }
            
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.warning(f"⚠️ Failed after {MAX_RETRIES} attempts: {str(e)}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

def generate_summary(text, title):
    """Safe summary generation with input validation"""
    try:
        if not text or len(text) < 100:
            return None
            
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            device=DEVICE
        )
        
        # Handle long texts by chunking
        max_chars = 10000  # Model limit
        text_chunk = text[:max_chars] if len(text) > max_chars else text
        
        result = summarizer(
            f"{title}\n\n{text_chunk}",
            max_length=SUMMARY_LENGTH,
            min_length=SUMMARY_LENGTH//2,
            do_sample=False
        )
        return result[0]['summary_text'] if result else None
        
    except Exception as e:
        logger.warning(f"⚠️ Summary failed: {str(e)}")
        return text[:SUMMARY_LENGTH] + "..." if text else None

def discover_news(topic, num_articles=MAX_ARTICLES, region="us", language="en"):
    """Main discovery function with enhanced reliability"""
    if not verify_serper_api():
        raise ConnectionError("Serper.dev API unavailable")

    payload = {
        'q': f"{topic} news",
        'num': num_articles,
        'gl': region,
        'hl': language
    }

    try:
        # Get news results with retry logic
        for attempt in range(MAX_RETRIES + 1):
            try:
                news_resp = requests.post(
                    'https://google.serper.dev/news',
                    headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
                    json=payload,
                    timeout=15
                )
                
                if news_resp.status_code == 401:
                    raise Exception("Invalid Serper.dev API key")
                elif news_resp.status_code == 403:
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    raise Exception("API quota exceeded")
                    
                news_resp.raise_for_status()
                news_items = news_resp.json().get('news', [])[:num_articles]
                break
                
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES:
                    raise
                logger.warning(f"🔄 Retry {attempt + 1}/{MAX_RETRIES}: {str(e)}")
                time.sleep(2 ** attempt)

        articles = []
        for idx, item in enumerate(news_items, 1):
            try:
                article_data = {
                    "id": idx,
                    "title": item.get('title'),
                    "url": item.get('link'),
                    "source": item.get('source'),
                    "date": item.get('date'),
                    "snippet": item.get('snippet', '')[:200],
                    "image_url": item.get('imageUrl'),
                    "topic": topic
                }

                # Enhanced content processing
                content = extract_article_content(item['link'])
                if content:
                    article_data.update(content)
                    article_data['summary'] = generate_summary(
                        content['full_text'],
                        article_data['title']
                    ) or article_data['snippet']
                    
                    articles.append(article_data)
                    logger.info(f"✅ Processed article {idx}/{num_articles}")
                    time.sleep(REQUEST_DELAY)

            except Exception as e:
                logger.warning(f"⚠️ Failed to process article {idx}: {str(e)}")

        return {
            "metadata": {
                "topic": topic,
                "region": region,
                "language": language,
                "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(articles),
                "status": "success"
            },
            "articles": articles
        }

    except Exception as e:
        logger.error(f"❌ News discovery failed: {str(e)}")
        return {
            "metadata": {
                "topic": topic,
                "error": str(e),
                "status": "failed"
            },
            "articles": []
        }

def save_results(data, filename="news_discovery.json"):
    """Save results with error handling"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved results to {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {str(e)}")

if __name__ == "__main__":
    try:
        topic = "latest developments in India"
        logger.info(f"🔍 Searching for: {topic}")
        
        results = discover_news(topic, num_articles=5, region="in")  # India region
        
        if results['metadata']['status'] == "success" and results['metadata']['count'] > 0:
            save_results(results)
            
            # Print sample output
            sample = results['articles'][0]
            print("\n=== SAMPLE ARTICLE ===")
            print(f"📰 Title: {sample['title']}")
            print(f"🏢 Source: {sample['source']}")
            print(f"📅 Date: {sample['date']}")
            print(f"\n📝 Summary ({len(sample.get('summary', ''))} chars):")
            print(sample.get('summary', 'No summary available'))
            print(f"\n🖼️ Image URL: {sample.get('image_url', sample.get('top_image', 'Not available'))}")
        else:
            logger.error("❌ No articles retrieved. Check logs for errors.")
            
    except Exception as e:
        logger.error(f"💥 Critical failure: {str(e)}")