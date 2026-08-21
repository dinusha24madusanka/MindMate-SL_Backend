from pathlib import Path
import json

import intent
import joblib
import stress

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

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

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# SAFETY TERMS
# Prototype safety gate

HIGH_RISK_TERMS = [
    # English
    "kill myself",
    "suicide",
    "end my life",
    "want to die",
    "hurt myself",
    "self harm",

    # Romanized Sinhala / Singlish
    "marenn ona",
    "marenda hithenawa",
    "marenna hithenawa",
    "mata marenna ona",
    "jeewithe epa",
    "jeewithe iwara karanna",

    # Sinhala
    "මැරෙන්න ඕන",
    "මැරෙන්න හිතෙනවා",
    "ජීවිතේ එපා"
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

        x = self.embedding(
            input_ids
        )

        x = x.transpose(
            1,
            2
        )

        x = self.conv(
            x
        )

        x = self.relu(
            x
        )

        x = self.pool(
            x
        )

        x = x.transpose(
            1,
            2
        )

        _, (
            hidden,
            _
        ) = self.lstm(
            x
        )

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

        cls.intent_model.to(
            DEVICE
        )

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

        cls.emotion_max_length = (
            emotion_config["max_length"]
        )

        cls.pad_index = (
            cls.emotion_word2idx["<PAD>"]
        )

        cls.unk_index = (
            cls.emotion_word2idx["<UNK>"]
        )

        cls.emotion_model = CNNLSTMClassifier(

            vocab_size=
                emotion_config["vocab_size"],

            embedding_dim=
                emotion_config["embedding_dim"],

            cnn_channels=
                emotion_config["cnn_channels"],

            lstm_hidden=
                emotion_config["lstm_hidden"],

            num_classes=
                emotion_config["num_classes"]
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

        inputs = {
            key: value.to(
                DEVICE
            )
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
            token_ids += [

                cls.pad_index

            ] * (

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

        cls.load_models()


        prediction = int(

            cls.stress_model.predict(
                [text]
            )[0]
        )


        probabilities = (
            cls.stress_model
            .predict_proba(
                [text]
            )[0]
        )


        classifier = (
            cls.stress_model
            .named_steps["classifier"]
        )


        classes = list(
            classifier.classes_
        )


        if 1 in classes:

            stress_index = (
                classes.index(1)
            )

            stress_probability = float(
                probabilities[
                    stress_index
                ]
            )

        else:

            stress_probability = 0.0


        # IMPORTANT:
        # This is model probability,
        # NOT a clinical stress severity score.
        stress_score = round(
            stress_probability
            * 100
        )
        stress_score = round(
            stress_probability * 100
        )

        # ==================================
        # Keyword assisted stress boost
        # ==================================

        stress_keywords = [
            "stress",
            "stressed",
            "pressure",
            "baya",
            "bayai",
            "hithenawa",
            "amarui",
            "dukai",
            "godak stress",
            "tension"
        ]

        normalized = text.lower()

        if any(
                word in normalized
                for word in stress_keywords
        ):

            if stress_score < 60:
                stress_score = 65


        return {
            "label_id": prediction,

            "label":
                (
                    "stress"
                    if prediction == 1
                    else "non_stress"
                ),

            "probability":
                stress_probability,

            "score":
                stress_score
        }

    # STRESS LEVEL

    @staticmethod
    def stress_level(score: int):

        if score >= 75:
            return "HIGH"

        if score >= 50:
            return "MODERATE"

        return "LOW"

    # ACTIVITY SELECTION

    @staticmethod
    def choose_activity(
            risk_level: str,
            stress_level: str,
            emotion: str
    ):

        if risk_level == "HIGH":
            return "SAFETY_SUPPORT"

        # HIGH STRESS
        if stress_level == "HIGH":

            if emotion in {
                "fear",
                "anxiety",
                "sadness",
                "anger"
            }:
                return "BREATHING"

            return "GROUNDING"

        # MODERATE STRESS
        if stress_level == "MODERATE":

            if emotion in {
                "fear",
                "anxiety",
                "sadness"
            }:
                return "MANDALA"

            return "CALM_BUBBLES"

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

        # ================================
        # SAFETY
        # ================================

        if risk_level == "HIGH":
            return (
                "Oya kiyapu de serious. "
                "Puluwan nam dan thaniyama inna epa. "
                "Oya trust karana kenek ekka inna. "
                "Professional support ekak ganna puluwan nam "
                "eka karanna. "
                "Me situation eke MindMate games suggest "
                "karanne naha."
            )

        # ================================
        # HIGH STRESS
        # ================================

        if stress_level == "HIGH":

            if language == "SINHALA":
                return (
                    "ඔයාගේ message එකෙන් දැන් පීඩනය වැඩි "
                    "වගේ මට තේරෙනවා. "
                    f"පොඩි වෙලාවක් අරගෙන {activity_name} "
                    "කරලා මනස ටිකක් සන්සුන් කරගන්න පුළුවන්. "
                    "ඊට පස්සේ ඔයාට තියෙන ප්‍රශ්නය ගැන "
                    "අපි කතා කරමු."
                )

            return (
                "Oyage message eken pressure eka wadi "
                "wage mata theruna. "
                f"Podiyak calm wenna {activity_name} "
                "try karanna puluwan. "
                "Passe api oyata thiyena de gena "
                "katha karamu."
            )

        # ================================
        # MODERATE STRESS
        # ================================

        if stress_level == "MODERATE":

            if intent in {
                "EXAM_STRESS",
                "EXAM",
                "STUDY_STRESS"
            }:
                return (
                    "Exam eka nisa oyata baya saha "
                    "pressure eka enawa wage mata theruna. "
                    "Eka normal deyak. "
                    "Podi break ekak aran calm wenna. "
                    "Oyata wada bayak enne mona part "
                    "ekatada kiyanna. "
                    "Api eka step by step balamu."
                )

            return (
                "Oyage message eken tikak pressure ekak "
                "penenawa wage. "
                f"Oya kemathi nam {activity_name} "
                "ekak karala podi break ekak ganna puluwan."
            )

        # ================================
        # LOW / NORMAL CHAT
        # ================================

        if intent in {
            "EXAM_STRESS",
            "EXAM"
        }:
            return (
                "Exam eka gena oyata podi bayak "
                "hithena eka mata theruna. "
                "Oyata amaruma wenne mona kotasatada "
                "kiyanna. "
                "Api eka solve karamu."
            )

        if intent in {
            "SADNESS",
            "LONELY"
        }:
            return (
                "Oyata tikak amarui wage mata theruna. "
                "Ehema hithenna hethuwa mokakda "
                "kiyala mata kiyanna puluwanda?"
            )

        if intent == "WORK_STRESS":
            return (
                "Weda walin pressure eka wadi wela wage. "
                "Stress wenne mona deyak nisa da "
                "api balamu."
            )

        return (
            "Oya kiyapu eka mata theruna. "
            "E gena tikak kiyanna puluwanda?"
        )

    # FINAL HYBRID ANALYSIS
    @classmethod
    def analyze(cls, text: str):
        text = str(text).strip()

        stress = cls.predict_stress(text)

        emotion = cls.predict_emotion(text)

        activity = cls.choose_activity(
            risk_level,
            stress["level"],
            emotion["label"]
        )
        if not text:
            raise ValueError(
                "Message cannot be empty."
            )
        # Safety gate MUST run first
        safety = cls.detect_safety_risk(
            text
        )
        # HIGH-RISK FLOW
        if safety["risk_level"] == "HIGH":
            reply = cls.generate_reply(
                text=text,
                intent="SAFETY",
                risk_level="HIGH",
                stress_level="NOT_EVALUATED",
                emotion="NOT_EVALUATED",
                activity="SAFETY_SUPPORT"
            )

            print("======================")
            print("DEBUG TEXT:", text)
            print("DEBUG INTENT:", intent)
            print("DEBUG EMOTION:", emotion)
            print("DEBUG STRESS:", stress)
            print("DEBUG RISK:", risk_level)
            print("DEBUG ACTIVITY:", activity)
            print("======================")

            return {
                "reply": reply,

                "intent": "SKIPPED",

                "intent_confidence": 0.0,

                "emotion": "SKIPPED",

                "emotion_confidence": 0.0,

                "stress_score": 0,

                "stress_probability": 0.0,

                "stress_level":
                    "NOT_EVALUATED",

                "risk_level": "HIGH",

                "allow_gamification": False,

                "recommended_activity":
                    "SAFETY_SUPPORT"
            }

        # NORMAL NLP FLOW
        intent = cls.predict_intent(
            text
        )


        emotion = cls.predict_emotion(
            text
        )


        stress = cls.predict_stress(
            text
        )


        stress_level = cls.stress_level(
            stress["score"]
        )


        activity = cls.choose_activity(

            risk_level=
                safety["risk_level"],

            stress_level=
                stress_level,

            emotion=
                emotion["label"]
        )

        reply = cls.generate_reply(
            text=text,
            intent=intent["label"],
            risk_level=safety["risk_level"],
            stress_level=stress_level,
            emotion=emotion["label"],
            activity=activity
        )


        return {
            "reply": reply,
            "intent":
                intent["label"],
            "intent_raw":
                intent["raw_label"],
            "intent_confidence":
                round(
                    intent["confidence"],
                    4),
            "emotion": emotion["label"],

            "emotion_confidence":round(
                    emotion["confidence"],
                    4),
            "stress_score": stress["score"],

            "stress_probability": round(stress["probability"],
                    4),

            "stress_level":
                stress_level,
            "risk_level":
                safety["risk_level"],
            "allow_gamification":
                safety[
                    "allow_gamification"
                ],
            "recommended_activity":
                activity
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
