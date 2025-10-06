from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    status: str
    chunks_added: int

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]