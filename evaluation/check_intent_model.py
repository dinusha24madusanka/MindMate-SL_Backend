from app.services import HybridNLPService

print("=" * 60)
print("MindMate-SL XLM-R Intent Model Check")
print("=" * 60)

HybridNLPService.load_models()

print("\nNumber of intent classes:")
print(len(HybridNLPService.intent_id2label))
print("\nIntent labels:")

for class_id, label in HybridNLPService.intent_id2label.items():
    print(class_id, "->", label)

print("\nTest Prediction:")
text = "mata exam eka gana godak stress"
result = HybridNLPService.predict_intent(text)

print("Text       :", text)
print("Prediction :", result)