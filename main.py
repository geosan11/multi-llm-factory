import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from anthropic import AsyncAnthropic

app = FastAPI(title="Multi-LLM Factory Bot")

# Initialize API clients safely using environment variables
@app.on_event("startup")
async def startup_event():
    app.state.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    app.state.claude_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class PromptPayload(BaseModel):
    prompt: str

@app.get("/")
def health_check():
    return {"status": "factory_online", "message": "Running on free tier architecture"}

@app.post("/route")
async def route_and_execute(payload: PromptPayload):
    prompt = payload.prompt
    
    # 1. Simple heuristic pre-routing rule
    if "code" in prompt.lower() or "refactor" in prompt.lower():
        # Route to Claude
        try:
            response = await app.state.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"engine": "claude", "response": response.content[0].text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Route everything else to Gemini Flash to stay light and fast
        try:
            response = app.state.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return {"engine": "gemini", "response": response.text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
