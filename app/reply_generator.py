import os

from google import genai


class DynamicReplyGenerator:

    _client = None

    MODEL_NAME = "gemini-3.1-flash-lite"


    @classmethod
    def _get_client(cls):

        if cls._client is None:

            api_key = os.getenv(
                "GEMINI_API_KEY"
            )

            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not configured."
                )

            cls._client = genai.Client(
                api_key=api_key
            )

        return cls._client


    @classmethod
    def generate(
        cls,
        *,
        text: str,
        language: str,
        intent: str,
        intent_raw: str,
        intent_confidence: float,
        emotion: str,
        emotion_confidence: float,
        stress_score: int,
        stress_level: str,
        recommended_activity: str
    ) -> str:

        prompt = f"""
You are MindMate-SL, a supportive wellbeing chatbot
for Sri Lankan university students.

Your job is to respond naturally to the USER MESSAGE,
not to repeat a fixed template.

USER MESSAGE:
{text}

NLP INFORMATION:
- detected language: {language}
- intent: {intent}
- raw intent prediction: {intent_raw}
- intent confidence: {intent_confidence:.4f}
- emotion: {emotion}
- emotion confidence: {emotion_confidence:.4f}
- model stress probability score: {stress_score}/100
- stress category: {stress_level}
- selected supportive activity: {recommended_activity}

IMPORTANT RULES:

1. The USER MESSAGE is the most important context.

2. Intent, emotion and stress values are model predictions.
Do not present them as medical facts.

3. If intent confidence is low or intent is UNCERTAIN,
do not assume the raw intent is correct.
Understand the user mainly from their actual message.

4. Reply in the same language style as the user:
- Sinhala Unicode -> Sinhala
- Romanized Sinhala / Singlish -> Romanized Sinhala
- English -> English

5. Give a warm, natural and context-specific response.

6. Do NOT simply say:
"I understand", "tell me more",
or another generic fallback without responding
to what the user actually said.

7. For LOW stress:
- respond naturally to the situation
- acknowledge the user's concern when appropriate
- ask at most one useful follow-up question

8. For MODERATE stress:
- acknowledge the specific situation
- give one simple practical coping suggestion
- keep the response calm and short

9. For HIGH stress that is NOT a safety-risk case:
- use calm, simple wording
- suggest one immediate non-clinical coping step
  such as slowing breathing, pausing, grounding,
  or breaking the problem into a smaller next step

10. The Android app separately displays the activity card.
Do not invent another game or activity.
Do not repeatedly advertise the activity.

11. Do not diagnose depression, anxiety disorders,
or other mental-health conditions.

12. Do not claim that an activity will definitely
reduce stress or cure a condition.

13. Do not prescribe medication or provide treatment.

14. Do not claim to be a therapist or doctor.

15. Keep the reply approximately 2-5 short sentences.
Avoid long paragraphs.

Return ONLY the reply that should be shown to the user.
"""

        client = cls._get_client()

        interaction = client.interactions.create(
            model=cls.MODEL_NAME,
            input=prompt
        )

        reply = (
            interaction.output_text
            or ""
        ).strip()

        if not reply:
            raise RuntimeError(
                "Dynamic reply generator returned an empty response."
            )

        return reply