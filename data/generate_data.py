import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_STUDENTS = 600


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


def generate_dataset(n_students: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_hours = rng.normal(loc=5.5, scale=2.2, size=n_students).clip(0, 12)
    attendance = rng.normal(loc=85, scale=11, size=n_students).clip(30, 100)
    assignment_score = rng.normal(loc=78, scale=14, size=n_students).clip(0, 100)
    previous_score = rng.normal(loc=76, scale=14, size=n_students).clip(0, 100)

    # Weighted composite: previous performance and assignments matter most,
    # attendance and study hours give a smaller but real boost.
    composite = (
        study_hours * 3.2
        + attendance * 0.35
        + assignment_score * 0.30
        + previous_score * 0.30
    )
    noise = rng.normal(loc=0, scale=1.5, size=n_students)
    composite = composite + noise

    # Standardize then remap onto a 0-100 band centered at 65 with a spread of 15,
    # so grade thresholds (40/55/70/85) produce a realistic bell-shaped mix of
    # grades instead of a flat min-max stretch (which crowded everyone into C/D).
    composite_z = (composite - composite.mean()) / composite.std()
    composite = (65 + composite_z * 15).clip(0, 100)

    grades = [grade_from_score(s) for s in composite]
    results = ["Pass" if g != "F" else "Fail" for g in grades]

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(1),
            "attendance": attendance.round(1),
            "assignment_score": assignment_score.round(1),
            "previous_score": previous_score.round(1),
            "grade": grades,
            "result": results,
        }
    )
    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv("data/students.csv", index=False)
    print(f"Wrote data/students.csv with {len(dataset)} rows")
    print(dataset["grade"].value_counts().sort_index())