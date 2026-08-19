from pydantic import BaseModel, Field


# =====================================================
# CHAT REQUEST
# =====================================================

class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User chat message"
    )


# =====================================================
# CHAT RESPONSE
# =====================================================

class ChatResponse(BaseModel):

    # Existing Android fields
    reply: str
    stress_score: int

    # Hybrid NLP fields
    intent: str

    intent_raw: str | None = None

    intent_confidence: float

    emotion: str

    emotion_confidence: float

    stress_probability: float

    stress_level: str

    risk_level: str

    allow_gamification: bool

    recommended_activity: str