"""
Generates a synthetic student performance dataset.

There's no real dataset provided for this project, so this script creates
a plausible one: four input features (study hours, attendance, assignment
score, previous score) combine into a weighted score, which is then noised
and binned into a Grade (A-F). Result (Pass/Fail) is derived from Grade.

Run:
    python data/generate_dataset.py
Produces:
    data/student_data.csv
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_STUDENTS = 1200


def grade_from_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    study_hours = np.clip(rng.normal(4.5, 2.0, N_STUDENTS), 0, 12).round(1)
    attendance = np.clip(rng.normal(80, 12, N_STUDENTS), 30, 100).round(0)
    assignment_score = np.clip(rng.normal(72, 15, N_STUDENTS), 0, 100).round(0)
    previous_score = np.clip(rng.normal(70, 15, N_STUDENTS), 0, 100).round(0)

    # Weighted composite with noise - this is what actually drives the grade.
    # Each feature is normalised to a 0-100 scale first, then blended, so the
    # weights below are genuine importances rather than arbitrary magnitudes.
    study_pct = np.clip(study_hours / 10 * 100, 0, 100)
    composite = (
        study_pct * 0.35
        + attendance * 0.25
        + assignment_score * 0.20
        + previous_score * 0.20
    )
    composite += rng.normal(0, 6, N_STUDENTS)
    composite = np.clip(composite, 0, 100)

    grades = np.array([grade_from_score(s) for s in composite])
    result = np.where(np.isin(grades, ["A", "B", "C", "D"]), "Pass", "Fail")

    df = pd.DataFrame(
        {
            "study_hours": study_hours,
            "attendance": attendance,
            "assignment_score": assignment_score,
            "previous_score": previous_score,
            "grade": grades,
            "result": result,
        }
    )

    out_path = "data/student_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["grade"].value_counts())


if __name__ == "__main__":
    main()