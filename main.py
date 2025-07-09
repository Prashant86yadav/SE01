from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from promptflow import PFClient
import os
import json
import traceback
from dotenv import load_dotenv

# Configuration
load_dotenv()
os.environ["PF_DISABLE_TRACING"] = "true"
os.environ["PROMPTFLOW_ENABLE_TRACING"] = "false"

app = FastAPI(title="PromptFlow API")

# CORS Setup (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_methods=["POST"],
    allow_headers=["*"],
)

class FlowInput(BaseModel):
    InputURL: str

@app.post("/run")
async def run_flow(input: FlowInput):
    """Execute PromptFlow with user input"""
    try:
        pf = PFClient()
        result = pf.test(flow="./", inputs={"InputURL": input.InputURL})
        
        # Log and return cleaned result
        print("✅ PF Result:", json.dumps(result, indent=2, default=str))
        return {
            "status": "success",
            "data": result,
            "llm_analysis": result.get("llm_analysis", {})  # Direct access to nested data
        }
    except Exception as e:
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"🔥 Error: {json.dumps(error_detail, indent=2)}")
        raise HTTPException(status_code=500, detail=error_detail)