import csv
import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.pkl")
HISTORY_PATH = "prediction_history.csv"
HISTORY_FIELDS = [
    "timestamp",
    "study_hours",
    "attendance",
    "assignment_score",
    "previous_score",
    "predicted_grade",
    "result",
]

PASSING_GRADES = {"A", "B", "C", "D"}


_bundle = joblib.load(MODEL_PATH)
model = _bundle["model"]
FEATURES = _bundle["features"]
MODEL_NAME = _bundle["model_name"]
MODEL_ACCURACY = _bundle["accuracy"]


def ensure_history_file():
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w", newline="") as f:
            csv.writer(f).writerow(HISTORY_FIELDS)


def append_to_history(row: dict):
    ensure_history_file()
    with open(HISTORY_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writerow(row)


@app.route("/")
def home():
    return render_template("index.html", model_name=MODEL_NAME, accuracy=MODEL_ACCURACY)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form

    try:
        values = {feature: float(data[feature]) for feature in FEATURES}
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please fill in all four fields with valid numbers."}), 400

    row_df = pd.DataFrame([values], columns=FEATURES)
    predicted_grade = model.predict(row_df)[0]
    result = "Pass" if predicted_grade in PASSING_GRADES else "Fail"

    append_to_history(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "study_hours": values["study_hours"],
            "attendance": values["attendance"],
            "assignment_score": values["assignment_score"],
            "previous_score": values["previous_score"],
            "predicted_grade": predicted_grade,
            "result": result,
        }
    )

    return jsonify({"grade": predicted_grade, "result": result})


@app.route("/history")
def history():
    ensure_history_file()
    df = pd.read_csv(HISTORY_PATH)
    records = df.iloc[::-1].to_dict(orient="records")  
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)