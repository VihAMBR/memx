from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from .system import MemorySystem

app = FastAPI(title="memx server")
memories = {}

def get_mem(user_id: str):
    if user_id not in memories:
        memories[user_id] = MemorySystem(user_id=user_id)
    return memories[user_id]

class AddMessage(BaseModel):
    user_id: str
    role: str
    content: str
    timestamp: str | None = None

class QueryMessage(BaseModel):
    user_id: str
    query: str
    top_k: int = 20

@app.post("/add")
def add_message(msg: AddMessage):
    mem = get_mem(msg.user_id)
    mem.add(msg.role, msg.content, msg.timestamp)
    return {"status": "ok"}

@app.post("/end_session")
def end_session(user_id: str):
    mem = get_mem(user_id)
    mem.end_session()
    return {"status": "ok"}

@app.post("/context")
def get_context(q: QueryMessage):
    mem = get_mem(q.user_id)
    return {"context": mem.get_context(q.query, q.top_k)}

@app.post("/answer")
def answer(q: QueryMessage):
    mem = get_mem(q.user_id)
    if mem._llm is None:
        raise HTTPException(status_code=400, detail="LLM not configured on this memory system")
    return {"answer": mem.answer(q.query, q.top_k)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("memx.server:app", host="0.0.0.0", port=8000, reload=True)
