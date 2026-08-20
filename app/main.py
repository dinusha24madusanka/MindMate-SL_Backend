from fastapi import FastAPI, HTTPException, status

from app.schemas import (
    ChatRequest,
    ChatResponse
)

from app.services import HybridNLPService


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(

    title="MindMate-SL Backend API",

    description=(
        "Hybrid NLP backend for MindMate-SL using "
        "XLM-RoBERTa intent_local_backup recognition, "
        "CNN-LSTM emotion classification, "
        "and stress classification."
    ),

    version="2.0.0"
)


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def read_root():

    return {
        "status": "running",
        "project": "MindMate-SL Backend",
        "version": "2.0.0",
        "nlp_engine": "Hybrid NLP"
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "MindMate-SL Backend",
        "models_loaded":
            HybridNLPService._loaded
    }


# =====================================================
# CHAT ANALYSIS
# =====================================================

@app.post(
    "/api/v1/chat/analyze",

    response_model=ChatResponse,

    status_code=status.HTTP_200_OK,

    summary="Analyze a MindMate-SL chat message"
)
async def analyze_chat_message(
    payload: ChatRequest
):

    try:

        result = (
            HybridNLPService.analyze(
                payload.message
            )
        )

        return ChatResponse(
            **result
        )


    except ValueError as error:

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=str(error)
        )


    except Exception as error:

        print(
            "Hybrid NLP API Error:",
            repr(error)
        )

        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=(
                "Unable to analyze the message "
                "at this time."
            )
        )