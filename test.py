import json
import re
import os
from promptflow.core import tool
import openai

# Configure Azure OpenAI from environment
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://<your-resource-name>.openai.azure.com/
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Helper: Repair common JSON issues
def repair_json(json_str: str) -> dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    # Basic fixes
    fixes = [
        (r'\(truncated[^)]*\)', ''),
        (r'\\"', '"'),
        (r'[\x00-\x1f]', ''),
        (r'\s+', ' ')
    ]
    for pat, repl in fixes:
        json_str = re.sub(pat, repl, json_str)
    if json_str.count('"') % 2:
        json_str += '"'
    if json_str.count('{') > json_str.count('}'):
        json_str += '}' * (json_str.count('{') - json_str.count('}'))
    try:
        return json.loads(json_str)
    except:
        # fallback extraction
        try:
            return {
                "headline": re.search(r'"headline":\s*"([^"]*)"', json_str).group(1),
                "cover_image": re.search(r'"cover_image":\s*"([^"]*)"', json_str).group(1),
                "body": re.search(r'"body":\s*"([\s\S]*?)"', json_str).group(1),
                "references": [ {"url": m.group(1), "summary": m.group(2)} for m in re.finditer(r'\{\s*"url":\s*"([^"]+)",\s*"summary":\s*"([^"]+)"', json_str) ]
            }
        except:
            return {"headline": None, "cover_image": None, "body": None, "references": []}

# Helper: Call Azure OpenAI and return structured JSON
def call_azure_openai(system_msg: str, user_prompt: str) -> dict:
    response = openai.ChatCompletion.create(
        engine=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=4000
    )
    content = response.choices[0].message.content
    return repair_json(content)

@tool
def beautify_article(input1) -> dict:
    """Transform LLM analysis into a polished article via Azure OpenAI"""
    # Unwrap list if passed as list
    data = input1[0] if isinstance(input1, list) and input1 else input1 or {}
    # Locate analysis payload
    raw = None
    out = data.get("output", {})
    raw = out.get("llm_analysis") or (out if any(k in out for k in ("title","article")) else None)
    if raw is None:
        raw = data.get("llm_analysis")
    # Parse or repair
    if isinstance(raw, str):
        parsed = repair_json(raw)
    elif isinstance(raw, dict):
        parsed = raw
    else:
        parsed = {}

    # Ensure basic keys
    headline = parsed.get("headline")
    cover_image = parsed.get("cover_image")
    body = parsed.get("body")
    references = parsed.get("references")

    # Build prompt
    system_msg = f"You are a professional editor. Create a full article based on the following data:"  
    user_prompt = (
        f"Title: {headline}\n"
        f"Cover Image: {cover_image}\n"
        f"Body: {body}\n"
        f"References: {json.dumps(references)}\n"
        "Output JSON with keys: title, cover_image, article, references."
    )

    # Call Azure OpenAI
    result = call_azure_openai(system_msg, user_prompt)

    # Fallback minimal
    return {
        "title": result.get("title", headline),
        "cover_image": result.get("cover_image", cover_image),
        "article": result.get("article", body),
        "references": result.get("references", references)
    }
