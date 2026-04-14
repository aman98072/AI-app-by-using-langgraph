from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import uuid
from datetime import datetime
from typing import Dict
from app.agent import response_generator

load_dotenv()

app = FastAPI(title="AI APP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str



# ─────────────────────────────────────────────────────────────────────────────
# NON-STREAMING ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(body: ChatRequest):
    try:
        return response_generator(body.message)
    except asyncio.TimeoutError:
        return {"error": "Generation took too long. Try /chat/async endpoint."}
    except Exception as e:
        return {"error": str(e)}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
