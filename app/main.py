from fastapi import FastAPI, status
from app.schemas import ChatRequest, ChatResponse
from app.services import NLPService

app = FastAPI(
    title="MindMate-SL Backend API",
    description="AI-driven stress analysis and response generator API for MindMate-SL",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {"status": "running", "project": "MindMate-SL Backend"}


@app.post(
    "/api/v1/chat/analyze",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze chat message for stress induction"
)
async def analyze_chat_message(payload: ChatRequest):
    """
    Android ඇප් එකෙන් එන පණිවිඩය ලබාගෙන NLP Service එක හරහා
    Stress Score සහ Reply එක ලබා දෙන ප්‍රධාන API Endpoint එක.
    """
    # Service එකට Text එක යවා ප්‍රතිඵල ලබා ගැනීම
    bot_reply, calculated_score = NLPService.analyze_stress_level(payload.message)

    return ChatResponse(reply=bot_reply, stress_score=calculated_score)