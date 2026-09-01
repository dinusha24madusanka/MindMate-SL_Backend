import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from app.services import HybridNLPService

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(
    BASE_DIR,
    "local_test_strict.csv"
)
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "xlmr_results"
)
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# SCENARIO ID -> CURRENT 12-CLASS XLM-R INTENT
SCENARIO_ID_TO_INTENT = {
    1: "ACADEMIC_STRESS",
    6: "ACADEMIC_STRESS",
    2: "CAMPUS_SAFETY_INJUSTICE",
    3: "CAMPUS_SAFETY_INJUSTICE",
    4: "FINANCIAL_STRESS",
    5: "DAILY_CAMPUS_LOGISTICS",
    18: "DAILY_CAMPUS_LOGISTICS",
    7: "SOCIAL_ISOLATION",
    8: "ACADEMIC_SUCCESS",
    9: "ACADEMIC_SUCCESS",
    10: "FINAL_PAPER_RELIEF",
    11: "FINAL_PAPER_RELIEF",
    12: "SOCIAL_EVENT",
    13: "SOCIAL_EVENT",
    14: "PERSONAL_WIN",
    15: "PERSONAL_WIN",
    16: "LECTURE_MANAGEMENT",
    17: "LECTURE_MANAGEMENT",
    19: "CAMPUS_RESOURCE_REQUEST",
    20: "CAMPUS_RESOURCE_REQUEST",
    21: "FOOD_AND_CANTEEN",
    22: "FOOD_AND_CANTEEN"
}

# LOAD TEST DATA
print("=" * 70)
print("MindMate-SL XLM-RoBERTa Final Evaluation")
print("=" * 70)
df = pd.read_csv(TEST_FILE)
print("\nTest samples:", len(df))
required_columns = [
    "Scenario_ID",
    "Clean_Text"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

# LOAD MODEL
print("\nLoading trained models...")
HybridNLPService.load_models()
print("\nXLM-R model loaded.")
print("Intent classes:",
    len(HybridNLPService.intent_id2label)
)

# EVALUATION
actual_labels = []
predicted_labels = []
confidences = []
app_labels = []
results = []

for index, row in df.iterrows():
    scenario_id = int(
        row["Scenario_ID"]
    )
    text = str(
        row["Clean_Text"]
    )
    actual = SCENARIO_ID_TO_INTENT.get(
        scenario_id
    )
    if actual is None:
        print(f"Skipping unknown Scenario_ID: "
            f"{scenario_id}"
        )
        continue
    prediction = (
        HybridNLPService.predict_intent(
            text
        )
    )
    raw_predicted = prediction["raw_label"]
    app_label = prediction["label"]
    confidence = float(prediction["confidence"])
    actual_labels.append(actual)
    predicted_labels.append(raw_predicted)
    app_labels.append(app_label)
    confidences.append(confidence)

    results.append({
        "Sample_ID":
            row.get(
                "Sample_ID",
                index
            ),
        "Scenario_ID": scenario_id,
        "Scenario_Name":
            row.get(
                "Scenario_Name",
                ""
            ),
        "Text": text,
        "Actual_Intent": actual,
        "Predicted_Intent": raw_predicted,
        "Application_Label": app_label,
        "Confidence": confidence,
        "Correct": actual == raw_predicted
    })
    if (index + 1) % 10 == 0:
        print(
            f"Processed "
            f"{index + 1}/{len(df)}"
        )

# METRICS
accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)
macro_precision = precision_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)
macro_recall = recall_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)
macro_f1 = f1_score(
    actual_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)
weighted_f1 = f1_score(
    actual_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)
uncertain_count = sum(
    1
    for label in app_labels
    if label == "UNCERTAIN"
)
uncertain_rate = (
    uncertain_count /
    len(app_labels)
)
average_confidence = (
    sum(confidences) /
    len(confidences)
)

# PRINT RESULTS

print("\n")
print("=" * 70)
print("FINAL XLM-R RESULTS")
print("=" * 70)
print(f"Test Samples : "
    f"{len(actual_labels)}"
)
print(f"Accuracy : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)
print(f"Macro Precision : "
    f"{macro_precision:.4f}"
)
print(
    f"Macro Recall : "
    f"{macro_recall:.4f}"
)
print(f"Macro F1-score : "
    f"{macro_f1:.4f} "
    f"({macro_f1 * 100:.2f}%)"
)
print(f"Weighted F1-score  : "
    f"{weighted_f1:.4f}"
)
print(f"Average Confidence : "
    f"{average_confidence:.4f}"
)
print(f"UNCERTAIN Count    : "
    f"{uncertain_count}"
)
print(f"UNCERTAIN Rate     : "
    f"{uncertain_rate * 100:.2f}%"
)

# SAVE PREDICTIONS
result_df = pd.DataFrame(
    results
)
prediction_file = os.path.join(
    OUTPUT_DIR,
    "xlmr_predictions.csv"
)
result_df.to_csv(
    prediction_file,
    index=False,
    encoding="utf-8-sig"
)

# CLASSIFICATION REPORT
report = classification_report(
    actual_labels,
    predicted_labels,
    output_dict=True,
    zero_division=0
)
report_df = (
    pd.DataFrame(
        report
    ).transpose()
)
report_file = os.path.join(
    OUTPUT_DIR,
    "xlmr_classification_report.csv"
)
report_df.to_csv(
    report_file,
    encoding="utf-8-sig"
)

# CONFUSION MATRIX
labels = sorted(
    set(actual_labels)
    | set(predicted_labels)
)

cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=labels
)

fig, ax = plt.subplots(
    figsize=(14, 12)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    values_format="d"
)

plt.title("MindMate-SL XLM-RoBERTa Intent Classification Confusion Matrix")
plt.tight_layout()
matrix_file = os.path.join(
    OUTPUT_DIR,
    "xlmr_confusion_matrix.png"
)
plt.savefig(
    matrix_file,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# SAVE SUMMARY
summary_file = os.path.join(
    OUTPUT_DIR,
    "xlmr_metrics.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:
    file.write("MindMate-SL XLM-RoBERTa Evaluation\n")
    file.write("=" * 50 + "\n")
    file.write(f"Test Samples: "
        f"{len(actual_labels)}\n"
    )
    file.write(f"Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)\n"
    )
    file.write(f"Macro Precision: "
        f"{macro_precision:.4f}\n"
    )
    file.write(f"Macro Recall: "
        f"{macro_recall:.4f}\n"
    )
    file.write(f"Macro F1: "
        f"{macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)\n"
    )
    file.write(f"Weighted F1: "
        f"{weighted_f1:.4f}\n"
    )
    file.write(f"Average Confidence: "
        f"{average_confidence:.4f}\n"
    )
    file.write(f"UNCERTAIN Count: "
        f"{uncertain_count}\n"
    )
    file.write(f"UNCERTAIN Rate: "
        f"{uncertain_rate * 100:.2f}%\n"
    )

print("\nFiles generated:")
print(prediction_file)
print(report_file)
print(matrix_file)
print(summary_file)
print("\nXLM-R evaluation COMPLETE.")