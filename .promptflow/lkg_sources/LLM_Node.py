from promptflow.core import tool
from typing import List, Dict
import os
import json
import asyncio
from openai import AsyncAzureOpenAI, APIConnectionError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Load environment variables
load_dotenv()

# --- JINJA SETUP ---
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['jinja'])
)
template = jinja_env.get_template("llm_instruction.jinja")

def find_cover_image(input1: List[Dict]) -> (str, str):
    """Find the first valid image_url and image_title across all items for cover image."""
    for item in input1:
        url = item.get("image_url")
        title = item.get("image_title")
        if url:
            return url, title
    return None, None

def render_instruction_jinja(context: dict) -> str:
    return template.render(context)

# --- AZURE OPENAI SETUP ---
def get_azure_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    if not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}"
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=endpoint,
        timeout=30.0
    )
client = get_azure_client()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def safe_llm_call(messages: list):
    try:
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1")
        return await client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
    except APIConnectionError as e:
        raise ConnectionError(f"Connection failed: {str(e)}")
    except APIStatusError as e:
        if e.status_code == 404:
            raise ValueError(
                f"Deployment not found. Please verify deployment '{deployment}' and endpoint."
            )
        raise

@tool
async def analyze_with_azure_openai(input1: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Processes all items together to generate one synthesized answer.
    """
    try:
        required_fields = ["title", "url", "snippet", "content", "favicon_url", "image_url", "image_title"]
        for item in input1:
            for k in required_fields:
                item.setdefault(k, "")

        cover_image_url, cover_image_title = find_cover_image(input1)
        context = {
            "input1": input1,
            "cover_image": cover_image_url,   # <--- renamed for consistency
            "cover_image_title": cover_image_title
        }
        system_message = render_instruction_jinja(context)
        messages = [
            {"role": "system", "content": system_message}
        ]
        response = await safe_llm_call(messages)
        llm_output = json.loads(response.choices[0].message.content)
        result = {
            "llm_analysis": llm_output,
            "llm_status": "success",
            "model_used": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        }
    except Exception as e:
        result = {
            "llm_analysis": {"error": str(e)},
            "llm_status": "Error",
            "model_used": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        }
    return result


# Local testing
if __name__ == "__main__":
    test_input = [
        {
            "url": "https://example.com",
            "title": "Sample Article",
            "snippet": "Sample snippet here...",
            "content": "This is a sample content to test the Azure OpenAI LLM pipeline.",
            "image_url": "https://example.com/image.jpg",
            "image_title": "Sample image",
            "favicon_url": "https://www.google.com/s2/favicons?domain=example.com"
        }
    ]
    results = asyncio.run(analyze_with_azure_openai(test_input))
    print(json.dumps(results, indent=2))
