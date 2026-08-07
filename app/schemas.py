from pydantic import BaseModel


class ChatResponse(BaseModel):
    reply: str
    stress_score: int


class ChatRequest(BaseModel):
    message: str