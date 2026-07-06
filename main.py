from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# 1. API එක නිර්මාණය කිරීම
app = FastAPI(title="MindMate-SL API")

# 2. XLM-RoBERTa මොඩලය Load කරගැනීම
print("Loading Model...")
classifier = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
print("Model Loaded!")


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    # මොඩලය හරහා හැඟීම් විශ්ලේෂණය
    result = classifier(request.message)
    label = result[0]['label']  # 'LABEL_0', 'LABEL_1', හෝ 'LABEL_2' ලෙස ලැබේ
    score = result[0]['score']  # 0.0 ත් 1.0 ත් අතර විශ්වාසනීයත්ව අගය (Confidence Score)

    # 3. ලැබෙන Label එක අනුව පිළිතුර සහ Stress Score එක තීරණය කිරීම
    if label == "LABEL_2":  # Positive (Low Stress)
        # score එක 1.0 ට ආසන්න වන විට stress score එක අඩු අගයක් (0-30) ගනී
        stress_score = int((1 - score) * 30)
        reply = "ඔයා අද හරිම සතුටින් වගේ! ඒක දකින්න ලැබීමත් සතුටක්."

    elif label == "LABEL_0":  # Negative (High Stress)
        # score එක 1.0 ට ආසන්න වන විට stress score එක වැඩි අගයක් (71-100) ගනී
        stress_score = int(71 + (score * 29))
        reply = "ඔයා ඉන්නේ ටිකක් ප්‍රශ්නෙකින් වගේ නේද? මම ඔයාට උදව් කරන්නම්."

    else:  # LABEL_1 -> Neutral (Medium Stress)
        stress_score = int(31 + (score * 39))
        reply = "හ්ම්ම්... ඔයා කියපු දේ මට වැටහුණා. තව විස්තර ටිකක් කියන්නකෝ."

    # Android ඇප් එක බලාපොරොත්තු වන JSON Response එක ආපසු යැවීම
    return {
        "reply": reply,
        "stress_score": stress_score
    }