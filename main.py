from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# 1. API එක නිර්මාණය කිරීම
app = FastAPI(title="MindMate-SL API")

# 2. XLM-RoBERTa මොඩල් එක ලෝඩ් කිරීම
# සටහන: අපි මෙතනදී sentiment-analysis එකක් කරමු.
print("Loading Model...")
classifier = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
print("Model Loaded!")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    # මොඩල් එකෙන් හැඟීම හඳුනාගැනීම
    result = classifier(request.message)
    label = result[0]['label']
    
    # ලොජික් එක - මොඩල් එක දෙන label එක අනුව අපේ උත්තරය තීරණය කිරීම
    if label == "positive":
        reply = "ඔයා අද හරිම සතුටින් වගේ! ඒක දකින්න ලැබීම සතුටක්."
    elif label == "negative":
        reply = "ඔයා ඉන්නේ ටිකක් දුකින් වගේ නේද? මම ඔයාට උදව් කරන්නම්."
    else:
        reply = "මම ඔයාගේ කතාව අහගෙන ඉන්නවා. කියන්න මොකද වෙන්නේ?"
        
    return {"reply": reply}