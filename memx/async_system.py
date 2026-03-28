from __future__ import annotations

import asyncio
from typing import Any
from .system import MemorySystem

class AsyncMemorySystem:
    def __init__(self, *args, **kwargs):
        self.mem = MemorySystem(*args, **kwargs)
        
    async def add(self, role: str, content: str, timestamp: str | None = None) -> None:
        return await asyncio.to_thread(self.mem.add, role, content, timestamp)
        
    async def end_session(self) -> None:
        return await asyncio.to_thread(self.mem.end_session)
        
    async def get_context(self, query: str, top_k: int = 20) -> str:
        return await asyncio.to_thread(self.mem.get_context, query, top_k)
        
    async def answer(self, question: str, top_k: int = 20) -> str:
        return await asyncio.to_thread(self.mem.answer, question, top_k)
        
    async def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.mem.search, query, top_k)
