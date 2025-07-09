from promptflow.core import tool
import os
import json
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Load environment variables
load_dotenv()

# Jinja2 Template Setup
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['jinja'])
)
template = jinja_env.get_template("beautification_instruction.jinja2")

# Azure OpenAI Setup
def get_azure_client():
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=endpoint,
        timeout=30.0
    )

@tool
async def beautify_article_llm(user_query: str, llm_output: dict) -> dict:
    """
    Beautifies and restructures an article for web display.
    llm_output: Should be the dictionary output from the previous node.
    """
    client = get_azure_client()

    # Cap the body/article length to ~10,000 characters
    body = llm_output.get("body", "")
    if len(body) > 10000:
        body = body[:10000] + "\n\n(Content truncated for length. Summarize or split into sections as needed.)"

    context = {
        "user_query": user_query,
        "headline": llm_output.get("headline"),
        "body": body,
        "cover_image": llm_output.get("cover_image"),
        "references": llm_output.get("references"),
    }
    system_message = template.render(context)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]

    response = await client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-35-turbo"),
        messages=messages,
        temperature=0.5,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# (Optional) Local test
if __name__ == "__main__":
    async def main():
        test_llm_output = {
            "headline": "Sample Title",
            "body": "A" * 12000,
            "cover_image": "https://example.com/sample.jpg",
            "references": []
        }
        user_query = "Sample user query"
        results = await beautify_article_llm(user_query, test_llm_output)
        print(json.dumps(results, indent=2))

    asyncio.run(main())