from pathlib import Path
import json
import math
import re
import threading
import joblib
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from app.reply_generator import DynamicReplyGenerator

# PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_INTENT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "intent"
)

# Public Hugging Face repository used by cloud deployments.
HF_INTENT_REPO = "Dinusha1234/MindMate-SL-XLMR-Intent"

# Prefer the local model during development. If the local folder is not
# present (for example on Render), Transformers will download/load the
# model directly from the public Hugging Face repository.
if LOCAL_INTENT_MODEL_DIR.exists():
    INTENT_MODEL_SOURCE = str(LOCAL_INTENT_MODEL_DIR)
    print(f"Intent model source: LOCAL -> {INTENT_MODEL_SOURCE}")
else:
    INTENT_MODEL_SOURCE = HF_INTENT_REPO
    print(f"Intent model source: HUGGING FACE -> {INTENT_MODEL_SOURCE}")

EMOTION_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "emotion"
)

STRESS_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "stress"
    / "stress_classifier.joblib"
)

# DEVICE
DEVICE = torch.device("cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# SAFETY TERMS
# Prototype safety gate
HIGH_RISK_TERMS = [
    # English
    "kill myself","suicide","end my life","want to die","hurt myself","self harm",
    # Romanized Sinhala / Singlish
    "marenn ona","marenda hithenawa","marenna hithenawa","mata marenna ona","jeewithe epa","jeewithe iwara karanna",
    # Sinhala
    "මැරෙන්න ඕන","මැරෙන්න හිතෙනවා","ජීවිතේ එපා"
]

# CNN-LSTM ARCHITECTURE
# Must match training architecture exactly
class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        cnn_channels: int,
        lstm_hidden: int,
        num_classes: int
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(
            kernel_size=2
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.dropout = nn.Dropout(
            0.30
        )
        self.classifier = nn.Linear(
            lstm_hidden,
            num_classes
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.transpose(1,2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1,2)
        _, (hidden,_) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(
            x
        )
        return self.classifier(
            x
        )

# HYBRID NLP SERVICE
class HybridNLPService:
    _loaded = False
    _load_lock = threading.Lock()
    EMOTION_RELIABILITY_THRESHOLD = 0.65
    intent_tokenizer = None
    intent_model = None
    intent_id2label = None
    emotion_model = None
    emotion_word2idx = None
    emotion_id2label = None
    emotion_max_length = None
    pad_index = None
    unk_index = None
    stress_model = None

    # LOAD ALL MODELS ONCE
    @classmethod
    def load_models(cls):
        if cls._loaded:
            return
        with cls._load_lock:
            if not cls._loaded:
                cls._load_models_unlocked()

    @classmethod
    def _load_models_unlocked(cls):
        if cls._loaded:
            return
        print("=" * 60)
        print("Loading MindMate-SL Hybrid NLP Models")
        print("=" * 60)
        print("Device:", DEVICE)

        # Intent — XLM-RoBERTa
        print("\nLoading Intent Model...")
        cls.intent_tokenizer = (
            AutoTokenizer.from_pretrained(
                INTENT_MODEL_SOURCE
            )
        )

        cls.intent_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                INTENT_MODEL_SOURCE
            )
        )

        cls.intent_model.to(DEVICE)
        cls.intent_model.eval()
        cls.intent_id2label = {
            int(key): value
            for key, value
            in cls.intent_model.config.id2label.items()
        }
        print("Intent model loaded")

        # Emotion — CNN-LSTM
        print("Loading Emotion Model...")
        with open(
            EMOTION_MODEL_DIR / "config.json",
            "r",
            encoding="utf-8"
        ) as file:
            emotion_config = json.load(
                file
            )
        with open(
            EMOTION_MODEL_DIR / "vocab.json",
            "r",
            encoding="utf-8"
        ) as file:
            cls.emotion_word2idx = json.load(
                file
            )
        cls.emotion_id2label = {
            int(key): value
            for key, value
            in emotion_config["id2label"].items()
        }
        cls.emotion_max_length = (emotion_config["max_length"])
        cls.pad_index = (cls.emotion_word2idx["<PAD>"])
        cls.unk_index = (cls.emotion_word2idx["<UNK>"])
        cls.emotion_model = CNNLSTMClassifier(
            vocab_size= emotion_config["vocab_size"],
            embedding_dim= emotion_config["embedding_dim"],
            cnn_channels= emotion_config["cnn_channels"],
            lstm_hidden= emotion_config["lstm_hidden"],
            num_classes= emotion_config["num_classes"]
        )

        cls.emotion_model.load_state_dict(
            torch.load(
                EMOTION_MODEL_DIR
                / "cnn_lstm_best.pt",
                map_location=DEVICE
            )
        )
        cls.emotion_model.to(
            DEVICE
        )
        cls.emotion_model.eval()
        print("Emotion model loaded")

        # Stress Classifier
        print("Loading Stress Model...")
        cls.stress_model = joblib.load(
            STRESS_MODEL_FILE
        )
        print("Stress model loaded")
        cls._loaded = True
        print("\nHybrid NLP models ready")
        print("=" * 60)

    # SAFETY GATE
    @staticmethod
    def detect_safety_risk(text: str):
        normalized = (
            str(text)
            .lower()
            .strip()
        )
        for term in HIGH_RISK_TERMS:
            if term in normalized:
                return {
                    "risk_level": "HIGH",
                    "allow_gamification": False,
                    "matched_term": term
                }

        return {
            "risk_level": "NONE",
            "allow_gamification": True,
            "matched_term": None
        }

    # INTENT
    @classmethod
    def predict_intent(cls, text: str):
        cls.load_models()
        inputs = cls.intent_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=96
        )

        inputs = {key: value.to(DEVICE)
            for key, value
            in inputs.items()
        }
        with torch.no_grad():
            outputs = cls.intent_model(
                **inputs
            )
            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )
            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )
        intent_id = int(
            prediction.item()
        )
        confidence_value = float(
            confidence.item()
        )
        raw_label = (
            cls.intent_id2label[
                intent_id
            ]
        )

        # XLM-R v2 confidence can be weak.
        # Do not expose uncertain prediction as truth.
        final_label = (
            raw_label
            if confidence_value >= 0.35
            else "UNCERTAIN"
        )
        return {
            "id": intent_id,
            "label": final_label,
            "raw_label": raw_label,
            "confidence": confidence_value
        }

    # EMOTION TEXT ENCODING
    @classmethod
    def encode_emotion_text(
        cls,
        text: str
    ):
        tokens = (
            str(text)
            .lower()
            .split()
        )
        token_ids = [
            cls.emotion_word2idx.get(
                token,
                cls.unk_index
            )
            for token
            in tokens[
                :cls.emotion_max_length
            ]
        ]
        if (
            len(token_ids)
            < cls.emotion_max_length
        ):
            token_ids += [cls.pad_index] * (
                cls.emotion_max_length
                - len(token_ids)
            )
        return torch.tensor(
            [token_ids],
            dtype=torch.long
        ).to(
            DEVICE
        )

    # EMOTION
    @classmethod
    def predict_emotion(
        cls,
        text: str
    ):
        cls.load_models()
        input_ids = (
            cls.encode_emotion_text(
                text
            )
        )
        with torch.no_grad():
            logits = cls.emotion_model(
                input_ids
            )
            probabilities = torch.softmax(
                logits,
                dim=1
            )
            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )
        emotion_id = int(
            prediction.item()
        )
        return {
            "id": emotion_id,
            "label":
                cls.emotion_id2label[
                    emotion_id
                ],
            "confidence":
                float(
                    confidence.item()
                )
        }

    # STRESS
    @classmethod
    def predict_stress(
        cls,
        text: str
    ):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized = " ".join(text.casefold().split())
        if not normalized:
            raise ValueError("text must not be empty")

        cls.load_models()
        if cls.stress_model is None:
            raise RuntimeError("stress model was not loaded")
        if not hasattr(cls.stress_model, "predict_proba"):
            raise TypeError("stress model must support predict_proba")

        probabilities = list(
            cls.stress_model.predict_proba([text])[0]
        )
        classes = list(
            getattr(cls.stress_model, "classes_", [])
        )
        if not classes:
            named_steps = getattr(
                cls.stress_model,
                "named_steps",
                {}
            )
            classifier = named_steps.get("classifier")
            classes = list(getattr(classifier, "classes_", [])
            )

        if len(classes) != len(probabilities):
            raise ValueError("stress model classes do not match its probabilities")
        if set(classes) != {0, 1}:
            raise ValueError("stress model must use binary classes 0 and 1")

        probability_values = [float(value) for value in probabilities]
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in probability_values
        ):
            raise ValueError("stress model returned invalid probabilities")
        stress_index = classes.index(1)
        stress_probability = probability_values[stress_index]
        prediction_index = max(
            range(len(probabilities)),
            key=probabilities.__getitem__
        )
        prediction = int(classes[prediction_index])

        # probability = raw stress-model probability
        # score = support-routing score used by MindMate
        # Neither value is a clinical assessment.
        model_probability = float(
            stress_probability
        )
        model_score = int(model_probability * 100)
        # Start with raw model score
        routing_score = model_score

        # CONSERVATIVE KEYWORD ASSISTANCE
        strong_stress_phrases = [
            # Singlish
            "godak stress",
            "godakma stress",
            "mara stress",
            "hari stress",
            "loku stress",
            "stress godak",
            "stress wadi",
            "stress eka wadi",
            "godak pressure",
            "loku pressure",
            "pressure eka wadi",
            "pressure eka unbearable",
            "baya hithenawa",
            "loku bayak",
            "hari bayai",
            "focus karaganna ba",
            "kisima deyakata focus karaganna ba",
            # English
            "very stressed",
            "extremely stressed",
            "too much stress",
            "a lot of pressure",
            "really overwhelmed",
            "completely overwhelmed",
            "cannot focus",
            "can't focus",
            # Sinhala
            "ගොඩක් ආතතිය",
            "ලොකු පීඩනය",
            "ගොඩක් බය",
        ]
        stress_terms = [
            # English
            "stress",
            "stressed",
            "pressure",
            "tension",
            "overwhelmed",
            "worried",
            "worry",
            "anxious",
            "afraid",
            # Singlish
            "bayai",
            "baya",
            "amarui",
            "hitha kalabala",
            # Sinhala
            "ආතතිය",
            "පීඩනය",
            "බය",
            "අමාරුයි",
        ]
        negations = {
            "not", "no", "never", "neither",
            "don't", "dont", "isn't", "isnt",
            "wasn't", "wasnt", "aren't", "arent",
            "weren't", "werent", "naha", "na", "ne"
        }
        neutral_phrases = {
            "blood pressure", "pressure cooker",
            "stress ball", "stress free", "stress-free",
            "stress management", "stress test", "stressed syllable"
        }

        def phrase_regex(phrase: str) -> re.Pattern[str]:
            phrase_pattern = re.escape(
                phrase.casefold()
            ).replace(r"\ ", r"\s+")
            return re.compile(
                rf"(?<!\w){phrase_pattern}(?!\w)",
                re.IGNORECASE
            )
        neutral_spans = [
            match.span()
            for neutral_phrase in neutral_phrases
            for match in phrase_regex(neutral_phrase).finditer(normalized)
        ]
        def contains_unnegated(phrase: str) -> bool:
            for match in phrase_regex(phrase).finditer(normalized):
                # Treat forms such as "stress-free" as non-stress wording.
                if normalized[match.end():].startswith("-free"):
                    continue
                if any(
                    start <= match.start() and match.end() <= end
                    for start, end in neutral_spans
                ):
                    continue
                prefix_words = re.findall(
                    r"[\w']+",
                    normalized[:match.start()]
                )[-3:]
                if not any(word in negations for word in prefix_words):
                    return True
            return False

        # Strong explicit stress wording
        if any(
                contains_unnegated(phrase)
                for phrase in strong_stress_phrases
        ):
            routing_score = max(
                routing_score,
                75
            )
        elif any(
                contains_unnegated(term)
                for term in stress_terms
        ):
            routing_score = max(
                routing_score,
                50
            )
        routing_score = max(0,
            min(
                routing_score,
                100
            )
        )
        return {
            "label_id": prediction,
            "label":
                (
                    "stress"
                    if prediction == 1
                    else "non_stress"
                ),
            # Raw classifier output
            "probability":
                model_probability,
            # Score used for support/activity routing
            "score":
                routing_score
        }

    # STRESS LEVEL
    @staticmethod
    def stress_level(score: int):
        if score >= 75:
            return "HIGH"
        if score >= 50:
            return "MODERATE"
        return "LOW"

    # POSITIVE MESSAGE GUARD
    @staticmethod
    def is_positive_message_without_distress(
        text: str
    ) -> bool:
        normalized = " ".join(
            str(text)
            .lower()
            .split()
        )
        positive_phrases = [
            # Singlish
            "mata sathutui",
            "godak sathutui",
            "hondatama giya",
            "hondin giya",
            "exam eka pass una",
            "pass una",
            "assignment eka submit kala",
            "submit kala",
            "complete kala",
            "iwara kala",
            "weda tika iwara una",
            # English
            "i am happy",
            "i'm happy",
            "went well",
            "it went well",
            "i passed",
            "passed my exam",
            "i finished",
            "i completed",
            "i submitted",
            "good news",
            "great day",
            # Sinhala
            "මට සතුටුයි",
            "ගොඩක් සතුටුයි",
            "හොඳට ගියා",
            "පාස් වුණා",
            "ඉවර කළා",
            "සම්පූර්ණ කළා",
        ]
        distress_terms = [
            # English
            "stress",
            "stressed",
            "pressure",
            "tension",
            "overwhelmed",
            "worried",
            "worry",
            "anxious",
            "afraid",
            # Singlish
            "baya",
            "bayai",
            "dukai",
            "amarui",
            "epa wela",
            "awul",
            "pressure eka",
            # Sinhala
            "ආතතිය",
            "පීඩනය",
            "බය",
            "දුකයි",
            "අමාරුයි",
        ]
        has_positive_context = any(
            phrase in normalized
            for phrase in positive_phrases
        )
        has_explicit_distress = any(
            term in normalized
            for term in distress_terms
        )
        return (
            has_positive_context
            and not has_explicit_distress
        )

    # ACTIVITY SELECTION
    @classmethod
    def choose_activity(
            cls,
            risk_level: str,
            stress_level: str,
            emotion: str
    ) -> str:
        # Safety always has highest priority
        if risk_level == "HIGH":
            return "SAFETY_SUPPORT"
        # High stress -> Mandala Paint Flow
        if stress_level == "HIGH":
            return "MANDALA"
        # Moderate stress -> Calm Bubbles
        if stress_level == "MODERATE":
            return "CALM_BUBBLES"
        # Low stress -> no activity
        return "NONE"

    # REPLY LANGUAGE DETECTION
    @staticmethod
    def detect_reply_language(text: str):
        text = str(text).strip()
        # Sinhala Unicode
        if any(
                "\u0D80" <= char <= "\u0DFF"
                for char in text
        ):
            return "SINHALA"

        # Romanized Sinhala / Singlish
        normalized = text.lower()
        singlish_markers = [
            "mata",
            "mama",
            "oya",
            "oyata",
            "api",
            "eka",
            "nisa",
            "heta",
            "ada",
            "godak",
            "wage",
            "hithenawa",
            "baya",
            "dukai",
            "karanna",
            "wenawa",
            "thiyenawa",
            "epa",
            "puluwan",
            "kemathi",
            "hari",
            "tikak"
        ]
        marker_count = sum(
            1
            for marker in singlish_markers
            if marker in normalized.split()
        )
        if marker_count >= 2:
            return "SINGLISH"
        # Default
        return "ENGLISH"

    # ACTIVITY DISPLAY NAME
    @staticmethod
    def activity_display_name(activity: str):
        names = {
            "GROUNDING": "5-4-3-2-1 Grounding",
            "BREATHING": "4-7-8 Breathing",
            "MANDALA": "Mandala Paint Flow",
            "CALM_BUBBLES": "Calm Bubbles",
            "NONE": "",
            "SAFETY_SUPPORT": ""
        }
        return names.get(
            activity,
            activity.replace("_", " ").title()
        )

    # LANGUAGE-ADAPTIVE RESPONSE GENERATOR
    @classmethod
    def generate_reply(
            cls,
            text: str,
            intent: str,
            risk_level: str,
            stress_level: str,
            emotion: str,
            activity: str
    ):
        language = cls.detect_reply_language(text)
        activity_name = cls.activity_display_name(activity)

        # HIGH-RISK SAFETY RESPONSE
        # Deterministic: Gemini is NOT used here.
        if risk_level == "HIGH":
            # Sinhala Unicode
            if language == "SINHALA":
                return (
                    "ඔයා කියපු දේ බරපතළයි. "
                    "පුළුවන් නම් මේ වෙලාවේ තනියම ඉන්න එපා. "
                    "ඔයා විශ්වාස කරන කෙනෙක් එක්ක ඉන්න හෝ "
                    "ඉක්මනින් සහාය ඉල්ලන්න. "
                    "සුදුසු මානසික සෞඛ්‍ය වෘත්තිකයෙකුගෙන් "
                    "වෘත්තීය සහාය ලබාගැනීමත් වැදගත්. "
                    "මේ අවස්ථාවේ MindMate ක්‍රීඩා හෝ "
                    "වෙනත් gamified activities යෝජනා කරන්නේ නැහැ."
                )

            # Romanized Sinhala / Singlish
            if language == "SINGLISH":
                return (
                    "Oya kiyapu de serious. "
                    "Puluwan nam me welawe thaniyama inna epa. "
                    "Oya trust karana kenek ekka inna hari "
                    "ikmanin support illanna. "
                    "Qualified mental health professional kenekgen "
                    "professional support ganna puluwan nam eka wedagath. "
                    "Me situation eke MindMate games saha "
                    "gamified activities suggest karanne naha."
                )
            # English
            return (
                "What you shared sounds serious. "
                "If possible, please do not stay alone right now. "
                "Stay with someone you trust or reach out for support as soon as you can. "
                "Seeking support from a qualified mental health professional is also important. "
                "MindMate will not suggest games or gamified activities in this situation."
            )

        # HIGH STRESS
        # TECHNICAL FALLBACK RESPONSE
        # Used only when Gemini generation fails.
        positive_context = (
            cls.is_positive_message_without_distress(
                text
            )
        )
        # POSITIVE / ACHIEVEMENT MESSAGE
        if positive_context:
            if language == "SINHALA":
                return (
                    "ඒක අහන්න ලැබීම සතුටක්, හොඳින් ගිය දේ ගැන තව ටිකක් කියන්න කැමති නම් මට කියන්න 🌿"
                )
            if language == "SINGLISH":
                return (
                    "Eka ahanna labuna eka sathutui, hondin giya de gana thawath kiyanna kemathi nam mata kiyanna 🌿"
                )
            return (
                "I'm glad to hear that, and you can share what went well if you'd like 🌿"
            )

        # HIGH SUPPORT-ROUTING STRESS
        if stress_level == "HIGH":
            if activity != "NONE":
                if language == "SINHALA":
                    return (
                        f"පීඩනය වැඩි වගේ නම් {activity_name} කරලා පොඩි විවේකයක් ගන්න 🌿"
                    )
                if language == "SINGLISH":
                    return (
                        f"Pressure eka wadi wage nam {activity_name} karala podi break ekak ganna 🌿"
                    )
                return (
                    f"Pressure feels heavy, so take a short break with {activity_name} 🌿"
                )

            # HIGH stress but activity intentionally suppressed
            if language == "SINHALA":
                return (
                    "පීඩනය වැඩි වෙලාවට පොඩි විවේකයක් අරගෙන හිතේ තියෙන දේ කියන්න 🌿"
                )
            if language == "SINGLISH":
                return (
                    "Pressure wadi welawe podi break ekak aran hithe thiyena de kiyanna 🌿"
                )
            return (
                "Pressure feels heavy, so take a short pause and share what's on your mind 🌿"
            )
        # MODERATE SUPPORT-ROUTING STRESS
        if stress_level == "MODERATE":
            if activity != "NONE":
                if language == "SINHALA":
                    return (
                        f"ටිකක් පීඩනයක් තියෙනවා නම් {activity_name} කරලා පොඩි විවේකයක් ගමුද? 🌿"
                    )
                if language == "SINGLISH":
                    return (
                        f"Tikak pressure nam {activity_name} karala podi break ekak gamuda? 🌿"
                    )
                return (
                    f"If stress feels heavy, try a short break with {activity_name} 🌿"
                )
            if language == "SINHALA":
                return (
                    "මේ පීඩනයට හේතුව ගැන තව ටිකක් කතා කරන්න කැමතිද? මම අහගෙන ඉන්නවා 🌿"
                )
            if language == "SINGLISH":
                return (
                    "Me pressure ekata hethuwa gana thawa tikak katha karanna kemathida? 🌿"
                )
            return (
                "If stress feels difficult, you can share a little more about what's causing it 🌿"
            )
        # LOW / NORMAL MESSAGE
        if language == "SINHALA":
            return (
                "ඔයා කියපු දේ මට තේරුණා, ඒ ගැන තව ටිකක් කියන්න කැමති නම් මම අහගෙන ඉන්නවා 🌿"
            )
        if language == "SINGLISH":
            return (
                "Oya kiyapu de mata theruna, e gana thawa katha karanna kemathi nam mata kiyanna 🌿"
            )
        return (
            "I understand what you shared, so feel free to tell me more if you'd like 🌿"
        )

    # FINAL HYBRID ANALYSIS
    @classmethod
    def analyze(cls, text: str):
        if text is None:
            raise ValueError("Message cannot be empty.")
        text = str(text).strip()
        if not text:
            raise ValueError("Message cannot be empty.")

        # SAFETY CHECK FIRST
        safety = cls.detect_safety_risk(text)

        # HIGH RISK SAFETY FLOW
        if safety["risk_level"] == "HIGH":
            reply = cls.generate_reply(
                text=text,
                intent="SAFETY",
                risk_level="HIGH",
                stress_level="NOT_EVALUATED",
                emotion="NOT_EVALUATED",
                activity="SAFETY_SUPPORT"
            )
            return {
                "reply": reply,
                "intent": "SKIPPED",
                "intent_raw": None,
                "intent_confidence": 0.0,
                "emotion": "SKIPPED",
                "emotion_confidence": 0.0,
                "stress_score": 0,
                "stress_probability": 0.0,
                "stress_level": "NOT_EVALUATED",
                "risk_level": "HIGH",
                "allow_gamification": False,
                "recommended_activity": "SAFETY_SUPPORT"
            }

        # Run the NLP models only after the safety gate.
        intent = cls.predict_intent(text)
        emotion = cls.predict_emotion(text)
        stress = cls.predict_stress(text)

        # CONFIDENCE-AWARE EMOTION
        if (
                emotion["confidence"]
                >= cls.EMOTION_RELIABILITY_THRESHOLD
        ):
            effective_emotion = emotion["label"]

        else:
            effective_emotion = "UNCERTAIN"

        # SAFETY RISK AND STRESS LEVEL ARE SEPARATE
        risk_level = safety["risk_level"]
        allow_gamification = safety["allow_gamification"]
        stress_level = cls.stress_level(stress["score"])

        # ACTIVITY SELECTION
        # POSITIVE MESSAGE GUARD
        positive_context = (
            cls.is_positive_message_without_distress(
                text
            )
        )

        # ACTIVITY SELECTION
        if positive_context:
            # Keep the original model outputs for
            # research/debugging, but do not show
            # an unnecessary wellbeing activity.
            activity = "NONE"
        else:
            activity = cls.choose_activity(
                risk_level=risk_level,
                stress_level=stress_level,
                emotion=effective_emotion
            )

        # AI RESPONSE
        language = cls.detect_reply_language(text)
        try:
            reply = DynamicReplyGenerator.generate(
                text=text,
                language=language,
                intent=intent["label"],
                intent_raw=intent["raw_label"],
                intent_confidence=intent["confidence"],
                emotion=effective_emotion,
                emotion_confidence=emotion["confidence"],
                stress_score=stress["score"],
                stress_probability=stress["probability"],
                stress_level=stress_level,
                recommended_activity=activity
            )
        except Exception as error:
            print(
                "Dynamic Reply Generator Error:",
                repr(error)
            )

            # Technical fallback only.
            # This is NOT the primary reply system.
            reply = cls.generate_reply(
                text=text,
                intent=intent["label"],
                risk_level=risk_level,
                stress_level=stress_level,
                emotion=effective_emotion,
                activity=activity
            )
        print(
            "MindMate analysis completed:",
            {
                "stress_level": stress_level,
                "risk_level": risk_level,
                "activity": activity
            }
        )
        return {
            "reply": reply,
            "intent": intent["label"],
            "intent_raw": intent["raw_label"],
            "intent_confidence": round(intent["confidence"],
                    4),
            "emotion": effective_emotion,
            "emotion_confidence": round(emotion["confidence"],
                    4),
            "stress_score": stress["score"],
            "stress_probability":round(stress["probability"],
                    4),
            "stress_level": stress_level,
            "risk_level": risk_level,
            "allow_gamification": allow_gamification,
            "recommended_activity": activity
        }

# BACKWARD COMPATIBILITY
# Keeps current main.py working until next step
class NLPService:
    @staticmethod
    def analyze_stress_level(
        text: str
    ) -> tuple[str, int]:
        result = (
            HybridNLPService.analyze(
                text
            )
        )
        return (
            result["reply"],
            result["stress_score"]
        )
