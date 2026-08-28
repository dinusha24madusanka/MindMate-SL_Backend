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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_FILE = os.path.join(
    BASE_DIR,
    "stress_test_strict.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "stress_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("MindMate-SL Stress Classifier Final Evaluation")
print("=" * 70)


# ---------------------------------------------------------
# Load test dataset
# ---------------------------------------------------------

df = pd.read_csv(TEST_FILE)

print("\nTest samples:", len(df))

if "text" not in df.columns:
    raise ValueError("Missing column: text")

if "label" not in df.columns:
    raise ValueError("Missing column: label")


texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)

actual = (
    df["label"]
    .astype(int)
    .tolist()
)


# ---------------------------------------------------------
# Load trained stress model
# ---------------------------------------------------------

print("\nLoading trained models...")

HybridNLPService.load_models()

stress_model = HybridNLPService.stress_model

print("Stress classifier loaded.")


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

predicted = stress_model.predict(
    texts
)

predicted = [
    int(x)
    for x in predicted
]


# ---------------------------------------------------------
# Probabilities
# ---------------------------------------------------------

probabilities = []

if hasattr(
    stress_model,
    "predict_proba"
):
    probs = stress_model.predict_proba(
        texts
    )

    for row in probs:
        probabilities.append(
            float(row[1])
        )

else:
    probabilities = [
        None
        for _ in predicted
    ]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    actual,
    predicted
)

precision = precision_score(
    actual,
    predicted,
    average="macro",
    zero_division=0
)

recall = recall_score(
    actual,
    predicted,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    actual,
    predicted,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    actual,
    predicted,
    average="weighted",
    zero_division=0
)


print("\n")
print("=" * 70)
print("FINAL STRESS CLASSIFIER RESULTS")
print("=" * 70)

print(
    f"Test Samples      : {len(actual)}"
)

print(
    f"Accuracy          : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Macro Precision   : "
    f"{precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Macro Recall      : "
    f"{recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"Macro F1-score    : "
    f"{f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print(
    f"Weighted F1-score : "
    f"{weighted_f1:.4f} "
    f"({weighted_f1 * 100:.2f}%)"
)


# ---------------------------------------------------------
# Predictions CSV
# ---------------------------------------------------------

result_df = pd.DataFrame({
    "Text": texts,
    "Actual": actual,
    "Predicted": predicted,
    "Stress_Probability": probabilities,
    "Correct": [
        a == p
        for a, p
        in zip(
            actual,
            predicted
        )
    ]
})

result_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "stress_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

report = classification_report(
    actual,
    predicted,
    target_names=[
        "Non-Stress",
        "Stress"
    ],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(
    report
).transpose().to_csv(
    os.path.join(
        OUTPUT_DIR,
        "stress_classification_report.csv"
    ),
    encoding="utf-8-sig"
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    actual,
    predicted,
    labels=[0, 1]
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Non-Stress",
        "Stress"
    ]
)

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "MindMate-SL Stress Classifier Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "stress_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Summary file
# ---------------------------------------------------------

with open(
    os.path.join(
        OUTPUT_DIR,
        "stress_metrics.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MindMate-SL Stress Classifier Evaluation\n"
    )

    file.write(
        "=" * 50 + "\n"
    )

    file.write(
        f"Test Samples: {len(actual)}\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)\n"
    )

    file.write(
        f"Macro Precision: "
        f"{precision:.4f}\n"
    )

    file.write(
        f"Macro Recall: "
        f"{recall:.4f}\n"
    )

    file.write(
        f"Macro F1: "
        f"{f1:.4f}\n"
    )

    file.write(
        f"Weighted F1: "
        f"{weighted_f1:.4f}\n"
    )


print("\nStress evaluation COMPLETE.")

print(
    "\nResults saved in:",
    OUTPUT_DIR
)