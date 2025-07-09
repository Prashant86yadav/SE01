from promptflow.core import tool
from urllib.parse import urlparse
import requests
from typing import List, Dict, Optional

API_KEY = "fb4c4d86127f32f8f6fd5b7cf3931e9c51997aa2"  # Replace/remove for prod!
SEARCH_ENDPOINT = "https://google.serper.dev/search"
IMAGES_ENDPOINT = "https://google.serper.dev/images"
TIMEOUT = 5.0
MAX_RESULTS = 7

def _validate_url(url: Optional[str]) -> Optional[str]:
    url = (url or "").strip()
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    try:
        parsed = urlparse(url)
        return url if all([parsed.scheme, parsed.netloc]) and len(url) <= 500 else None
    except:
        return None

def get_favicon_url(page_url):
    domain = urlparse(page_url).netloc
    return f"https://www.google.com/s2/favicons?domain={domain}"

@tool
def google_search_node(query: str) -> List[Dict[str, str]]:
    try:
        # Get organic results
        resp = requests.post(
            SEARCH_ENDPOINT,
            json={"q": query, "num": MAX_RESULTS},
            headers={"X-API-KEY": API_KEY, "Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()

        organic_results = [
            {
                "url": _validate_url(item.get("link")),
                "title": (item.get("title") or "")[:200],
                "snippet": (item.get("snippet") or "")[:1000],
                "position": i + 1,
                "favicon_url": get_favicon_url(item.get("link"))
            }
            for i, item in enumerate(data.get("organic", []))
            if item.get("link") and _validate_url(item.get("link"))
        ][:MAX_RESULTS]

        # Get images from Serper /images endpoint
        img_resp = requests.post(
            IMAGES_ENDPOINT,
            json={"q": query, "num": MAX_RESULTS},
            headers={"X-API-KEY": API_KEY, "Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        img_resp.raise_for_status()
        images_data = img_resp.json().get("images", [])

        # Attach images by position (1st img to 1st result, etc.)
        for i, item in enumerate(organic_results):
            img = images_data[i] if i < len(images_data) else None
            item["image_url"] = img.get("imageUrl") if img else None
            item["image_title"] = img.get("title") if img else None
            item["image_source"] = img.get("source") if img else None

        return organic_results

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Search failed: {str(e)}")
