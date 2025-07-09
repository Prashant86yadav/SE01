from promptflow.core import tool
from bs4 import BeautifulSoup
from typing import Dict, List, Union, Optional
import requests
from urllib.parse import urljoin, urlparse
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# --- CONFIG ---
MAX_CONTENT_LEN = 2000  # Limit content field to 10,000 characters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_favicon_url(page_url: str) -> str:
    domain = urlparse(page_url).netloc
    return f"https://www.google.com/s2/favicons?domain={domain}"

class EnterpriseWebCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9"
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_url(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                return response.text
            return None
        except Exception as e:
            logger.warning(f"Fetch failed (attempt {self.fetch_url.retry.statistics['attempt_number']}): {str(e)}")
            raise

    def extract_content(self, html: str, base_url: str) -> tuple[str, Optional[Dict[str, str]]]:
        soup = BeautifulSoup(html, 'lxml')
        for noise in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            noise.decompose()

        content_selectors = [
            ('article', 3),  # Highest priority
            ('main', 3),
            ('[role="main"]', 2),
            ('.article-body', 2),
            ('.post-content', 2),
            ('#content', 1),
            ('body', 0)      # Fallback
        ]

        best_node = None
        best_score = -1
        for selector, score in content_selectors:
            node = soup.select_one(selector)
            if node:
                text_length = len(node.get_text(strip=True))
                if text_length > 100 and score > best_score:
                    best_node = node
                    best_score = score
                    if score == 3:
                        break

        content = best_node.get_text(separator='\n', strip=True) if best_node else ""
        image = self.extract_first_image(soup, base_url)
        return content, image

    def extract_first_image(self, soup: BeautifulSoup, base_url: str) -> Optional[Dict[str, str]]:
        img_candidates = []
        # Meta tags
        meta_props = ['og:image', 'twitter:image', 'image_src']
        for prop in meta_props:
            for meta in soup.find_all('meta', attrs={'property': prop}):
                if url := self.normalize_url(meta.get('content', ''), base_url):
                    img_candidates.append({
                        "url": url,
                        "alt": "Featured image",
                        "source": "meta"
                    })
            for meta in soup.find_all('meta', attrs={'name': prop}):
                if url := self.normalize_url(meta.get('content', ''), base_url):
                    img_candidates.append({
                        "url": url,
                        "alt": "Featured image",
                        "source": "meta"
                    })

        # <link rel="image_src">
        for link in soup.find_all('link', rel="image_src"):
            if url := self.normalize_url(link.get('href', ''), base_url):
                img_candidates.append({
                    "url": url,
                    "alt": "Featured image",
                    "source": "link"
                })

        # <img> in main content and whole page
        for img in soup.find_all('img', src=True):
            src = img['src']
            if not (url := self.normalize_url(src, base_url)):
                continue
            alt = img.get('alt', '').strip() or None
            if any(x in src.lower() for x in ['sprite', 'spacer', 'blank', 'tracker', 'logo', 'icon', 'ads', '.svg', '.gif']):
                continue
            img_candidates.append({
                "url": url,
                "alt": alt,
                "source": "content"
            })

        # <picture> and <figure>
        for pic in soup.find_all('picture'):
            for source in pic.find_all('source', srcset=True):
                if url := self.normalize_url(source['srcset'].split(',')[0], base_url):
                    img_candidates.append({
                        "url": url,
                        "alt": None,
                        "source": "picture"
                    })

        for fig in soup.find_all('figure'):
            img = fig.find('img', src=True)
            if img:
                if url := self.normalize_url(img['src'], base_url):
                    img_candidates.append({
                        "url": url,
                        "alt": img.get('alt', '').strip() or None,
                        "source": "figure"
                    })

        # Validate and return the first valid image only
        def is_valid_image(url):
            try:
                resp = self.session.head(url, allow_redirects=True, timeout=5)
                ctype = resp.headers.get("Content-Type", "")
                clen = int(resp.headers.get("Content-Length", "0"))
                return ctype.startswith("image/") and clen > 2000
            except Exception:
                return False

        for img in img_candidates:
            if is_valid_image(img['url']):
                return img
        return None

    def normalize_url(self, url: str, base_url: str) -> Optional[str]:
        url = url.strip()
        if url.startswith('//'):
            url = f'https:{url}'
        elif url.startswith('/'):
            url = urljoin(base_url, url)
        return url if url.startswith('http') and len(url) < 1000 else None

    def process_item(self, item: Dict[str, str]) -> Dict[str, str]:
        url = item.get("url", "").strip()
        google_snippet = item.get("snippet", "")[:1000]
        favicon_url = get_favicon_url(url)

        if not url.startswith(('http://', 'https://')):
            return {
                **item,
                "content": "",
                "image": None,
                "favicon_url": favicon_url,
                "status": "error",
                "error": "Invalid URL",
                "source": "invalid_url"
            }

        try:
            if html := self.fetch_url(url):
                content, image = self.extract_content(html, url)
                if len(content) > 300:
                    return {
                        **item,
                        "content": content[:MAX_CONTENT_LEN],  # LIMIT HERE
                        "image": image,
                        "favicon_url": favicon_url,
                        "status": "success",
                        "error": "",
                        "source": "bs4"
                    }
            return {
                **item,
                "content": google_snippet,
                "image": None,
                "favicon_url": favicon_url,
                "status": "fallback",
                "error": "Using Google snippet",
                "source": "google_fallback"
            }
        except Exception as e:
            logger.error(f"Processing failed for {url}: {str(e)}")
            return {
                **item,
                "content": google_snippet,
                "image": None,
                "favicon_url": favicon_url,
                "status": "error",
                "error": f"Processing failed: {str(e)}",
                "source": "error_fallback"
            }

@tool
def crawl_url(item: Union[Dict[str, str], List[Dict[str, str]]]) -> Union[Dict[str, str], List[Dict[str, str]]]:
    crawler = EnterpriseWebCrawler()
    if isinstance(item, list):
        return [crawler.process_item(it) for it in item]
    return crawler.process_item(item)
