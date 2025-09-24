import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Basic AI Chatbot", version="1.0")

# Allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def index():
    return FileResponse("templates/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Single-turn: no chat history sent—just the user's one message
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": req.message},
            ],
            temperature=0.6,
            max_tokens=200,
        )
        reply = completion.choices[0].message.content.strip()
        return ChatResponse(reply=reply or "…")
    except Exception as e:
        # Minimal, safe error surfacing
        raise HTTPException(status_code=500, detail=f"Chat failed: {type(e).__name__}")