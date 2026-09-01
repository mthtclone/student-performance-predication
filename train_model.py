"""
Trains Decision Tree and Random Forest classifiers on the student
performance dataset, evaluates both with accuracy + a confusion matrix,
and saves the better-performing model for the Flask app to load.

Run from the project root:
    python model/train_model.py
Produces:
    model/model.pkl                 - the saved best model + feature/label metadata
    model/confusion_matrix.png      - confusion matrix for the best model
    model/metrics.txt               - accuracy + report for both models
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["study_hours", "attendance", "assignment_score", "previous_score"]
TARGET = "grade"
GRADE_ORDER = ["A", "B", "C", "D", "F"]

DATA_PATH = "data/student_data.csv"
MODEL_PATH = "model/model.pkl"
CONFUSION_MATRIX_PATH = "model/confusion_matrix.png"
METRICS_PATH = "model/metrics.txt"


def train_and_evaluate(name, clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, labels=GRADE_ORDER, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=GRADE_ORDER)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    return {
        "name": name,
        "model": clf,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = [
        train_and_evaluate(
            "Decision Tree",
            DecisionTreeClassifier(max_depth=6, random_state=42),
            X_train, X_test, y_train, y_test,
        ),
        train_and_evaluate(
            "Random Forest",
            RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
            X_train, X_test, y_train, y_test,
        ),
    ]

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nBest model: {best['name']} (accuracy {best['accuracy']:.4f})")

    # Save the best model plus everything the app needs to use it correctly.
    joblib.dump(
        {
            "model": best["model"],
            "model_name": best["name"],
            "features": FEATURES,
            "grade_order": GRADE_ORDER,
            "accuracy": best["accuracy"],
        },
        MODEL_PATH,
    )

    # Confusion matrix plot for the best model.
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=best["confusion_matrix"], display_labels=GRADE_ORDER)
    disp.plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"{best['name']} - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)

    # Plain-text metrics file for both models, useful for the README / reports.
    with open(METRICS_PATH, "w") as f:
        for r in results:
            f.write(f"=== {r['name']} ===\n")
            f.write(f"Accuracy: {r['accuracy']:.4f}\n")
            f.write(r["report"])
            f.write("\nConfusion matrix (rows=actual, cols=predicted), labels order "
                    f"{GRADE_ORDER}:\n")
            f.write(json.dumps(r["confusion_matrix"].tolist()))
            f.write("\n\n")
        f.write(f"Best model saved: {best['name']} (accuracy {best['accuracy']:.4f})\n")

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()