# Student Performance Predictor

Predicts a student's grade (A/B/C/D/F) from four inputs — study hours,
attendance, assignment score, and previous score — using a Random Forest
classifier (chosen over a Decision Tree after evaluation), and keeps a CSV
history of every prediction made through the web form.

## Project structure

```
app.py                     Flask app: serves the form, /predict, and /history
requirements.txt

data/
  generate_dataset.py      Builds the synthetic training dataset
  student_data.csv         Generated dataset (1200 rows)

model/
  train_model.py           Trains + evaluates Decision Tree and Random Forest
  model.pkl                Saved best model + feature/label metadata
  confusion_matrix.png     Confusion matrix for the best model
  metrics.txt              Accuracy + classification report for both models

templates/
  index.html               Predictor form + result
  history.html             Prediction history table

static/
  css/style.css
  js/script.js              Submits the form via fetch, renders the result

prediction_history.csv     Created automatically the first time a prediction is made
```

## Setup

```bash
pip install -r requirements.txt
```

## Train (or retrain) the model

Only needed once, or whenever you want to regenerate the dataset / retrain:

```bash
python data/generate_dataset.py
python model/train_model.py
```

This prints accuracy and a classification report for both Decision Tree and
Random Forest, saves a confusion matrix image, and stores the better model
to `model/model.pkl` for the app to load.

## Run the app

```bash
python app.py
```

Visit `http://127.0.0.1:5000/`. Fill in the form to get a prediction —
each one is appended to `prediction_history.csv` and shown on the
`/history` page.

## Notes on the dataset

There was no real dataset supplied for this project, so `generate_dataset.py`
creates a synthetic one: it builds a weighted composite of the four features
plus noise, then bins that composite into a grade. This keeps the features
genuinely predictive (not random) while leaving realistic overlap between
grades: the current Random Forest model gets about 69% test accuracy. If
you have a real dataset, replace `data/student_data.csv` with columns
`study_hours, attendance, assignment_score, previous_score, grade` and rerun
`model/train_model.py`.

---

# Student Performance Predictor Documentation

This document walks through every part of the project: where the data comes
from, how the model is trained and evaluated, how the Flask app serves
predictions, and how the frontend is wired together. It's meant to be a
reference for anyone picking the project back up later (including future-you).

---

## 1. What the project does

Given four inputs about a student 

- **Study hours** (per day)
- **Attendance** (%)
- **Assignment score** (0–100)
- **Previous score** (0–100)

It predicts a letter **Grade** (A/B/C/D/F), and derives a **Pass/Fail
Result** from that grade (A–D = Pass, F = Fail). Every prediction made
through the web form is logged to a CSV, and a `/history` page lets you
browse past predictions.

---

## 2. Project structure

```
app.py                      Flask app — serves pages, handles predictions
requirements.txt

data/
  generate_dataset.py        Builds the synthetic training dataset
  student_data.csv           The generated dataset (1200 rows)

model/
  train_model.py             Trains + evaluates both models, saves the best one
  model.pkl                  Saved model + metadata (loaded by app.py)
  confusion_matrix.png       Confusion matrix for the best model
  metrics.txt                Accuracy + full classification report, both models

templates/
  index.html                 Predictor form + result
  history.html                Prediction history table

static/
  css/style.css               All styling
  js/script.js                 Form submission + result rendering

prediction_history.csv       Log of every prediction (created on first use)
```

---

## 3. The dataset (`data/generate_dataset.py`)

No real student dataset was provided, so this script generates a synthetic
one that behaves like real data would: the features genuinely drive the
grade, but with enough noise and overlap between grade boundaries that no
model gets a suspiciously perfect score.

**How a student's grade is generated:**

1. Each of the four features is sampled from a normal distribution and
   clipped to a sensible range (e.g. attendance clipped to 30–100%).
2. `study_hours` is rescaled to a 0–100 "percent" scale (`study_hours / 10 *
   100`) so it can be combined with the other 0–100-scale features on equal
   footing.
3. A **composite score** is built as a weighted average:

   | Feature | Weight |
   |---|---|
   | Study hours (rescaled) | 0.35 |
   | Attendance | 0.25 |
   | Assignment score | 0.20 |
   | Previous score | 0.20 |

   Study hours is weighted highest deliberately — it's the feature most
   directly under a student's control, and gets the most causal weight.
4. Gaussian noise (`std=6`) is added to the composite, to mimic the fact
   that real academic performance always has an unpredictable component
   (a bad day, a hard exam, etc.), then the whole thing is clipped to 0–100.
5. The noised composite is binned into a grade:

   | Composite score | Grade |
   |---|---|
   | ≥ 85 | A |
   | 70–84 | B |
   | 55–69 | C |
   | 40–54 | D |
   | < 40 | F |

6. `result` is derived directly from `grade`: A/B/C/D → `Pass`, F → `Fail`.

Running the script writes `data/student_data.csv` with 1200 rows and these
six columns: `study_hours, attendance, assignment_score, previous_score,
grade, result`. The current class balance is roughly: C (52%), B (26%),
D (18%), A (2%), F (1%) — which is realistic (most students cluster in the
middle, few fail outright, few are exceptional) but means the model will
naturally be much better at predicting C/B/D than the rarer A/F.

**If you get a real dataset**, drop it in as `data/student_data.csv` with
the same column names and rerun `model/train_model.py` — nothing else needs
to change.

---

## 4. Training and evaluation (`model/train_model.py`)

This script is the one that satisfies the "Decision Tree / Random Forest"
and "Accuracy, Confusion Matrix" requirements directly.

**What it does, step by step:**

1. Loads `data/student_data.csv`.
2. Splits it 80/20 into train/test sets, **stratified by grade** — this
   matters because grades are imbalanced (only ~1% F), so a plain random
   split could leave the test set with zero F examples. Stratifying keeps
   the same class proportions in both splits.
3. Trains two models on the training set:
   - `DecisionTreeClassifier(max_depth=6)`
   - `RandomForestClassifier(n_estimators=200, max_depth=8)`

   Both depths are capped to avoid overfitting on a relatively small,
   noisy dataset — an unbounded tree would memorize training noise rather
   than learn the real signal.
4. For each model, evaluates on the held-out test set with:
   - **Accuracy** — overall fraction of correct predictions.
   - **Classification report** — precision/recall/F1 per grade, which
     matters more than accuracy alone here since the classes are
     imbalanced (a model could get ~50% accuracy just by always guessing
     "C").
   - **Confusion matrix** — a 5×5 grid of actual vs. predicted grade,
     showing exactly which grades get confused with which.
5. Picks whichever model has the higher test accuracy — currently the
   **Random Forest**, at about **69%** — and saves it.

**What gets saved:**

- `model/model.pkl` — a dictionary (via `joblib.dump`) containing:
  - `model` — the actual fitted scikit-learn estimator
  - `model_name` — `"Random Forest"` or `"Decision Tree"`, for display
  - `features` — the exact feature order the model expects
  - `grade_order` — `["A", "B", "C", "D", "F"]`, used for the confusion
    matrix axis labels
  - `accuracy` — the test accuracy, shown in the app's header
- `model/confusion_matrix.png` — a plotted confusion matrix for the
  winning model (rows = actual grade, columns = predicted grade; darker
  cells = more students).
- `model/metrics.txt` — accuracy and the full classification report for
  **both** models, plus the raw confusion matrix numbers, so you can
  compare Decision Tree vs. Random Forest side by side without rerunning
  anything.

**Reading the confusion matrix:** the diagonal is where the model got it
right. Off-diagonal cells just above or below the diagonal (e.g. B
predicted as C) are the model confusing adjacent grades expected, since
adjacent grade boundaries are the noisiest part of the data by
construction. Cells far from the diagonal (e.g. A predicted as F) would be
a red flag, but shouldn't really occur here given how the data was
generated.

Bundling `features` and `grade_order` into the same file as the model
means `app.py` never has to guess the feature order or hardcode grade
labels: it just reads them out of the same file it loads the model from,
so training and serving can't drift out of sync.

---

## 5. The Flask app (`app.py`)

### Startup

On import, the app loads `model/model.pkl` once and keeps the model,
feature list, model name, and accuracy in memory — so a prediction never
touches disk beyond appending to the history CSV.

### Routes

**`GET /`** — renders `templates/index.html`, passing in `model_name` and
`accuracy` so the page can show something like "Model: Random Forest ·
Test accuracy 69%" without hardcoding it in the template.

**`POST /predict`** — the core prediction endpoint.

- Accepts either JSON (`Content-Type: application/json`) or a regular form
  post; `request.get_json(silent=True) or request.form` handles both.
- Reads the four features **in the exact order stored in `model.pkl`**
  (`FEATURES`), converting each to `float`. If any field is missing or
  isn't a valid number, it returns `400` with a plain-language error
  message rather than a stack trace.
- Wraps the values in a single-row `pandas.DataFrame` (scikit-learn models
  expect a 2D input, and using a DataFrame with the training column names
  avoids "feature name mismatch" warnings from scikit-learn).
- Calls `model.predict(...)`, derives `result` from whether the predicted
  grade is in `{"A", "B", "C", "D"}`.
- Appends a row to `prediction_history.csv` with a timestamp, the four
  inputs, the predicted grade, and the result.
- Returns JSON: `{"grade": "B", "result": "Pass"}`.

**`GET /history`** — reads `prediction_history.csv` with pandas, reverses
the row order (newest first), and renders `templates/history.html` with
the records. If the file doesn't exist yet (no predictions made), it's
created with just the header row first, so the page always renders
cleanly with an empty-state message instead of erroring.

### `prediction_history.csv`

This is the "prediction history" extension from the requirements. It's a
plain CSV with columns:

```
timestamp, study_hours, attendance, assignment_score, previous_score, predicted_grade, result
```

It's created lazily (`ensure_history_file`) the first time it's needed —
either on the first prediction or the first visit to `/history` — so
there's nothing to set up manually. It grows by one row per prediction and
is otherwise just append-only; nothing rewrites past rows.

---

## 6. The frontend

### `templates/index.html`

The form has `id="predict-form"` and each input has a `name` matching the
feature name (`study_hours`, `attendance`, `assignment_score`,
`previous_score`) — this is what lets `script.js` read the values and post
them straight through to `/predict` without any relabeling.

The result side (`.stamp`) has a big `#prediction` element that starts as
an em dash and gets filled in with the predicted letter after the first
successful prediction, plus a row of five circular "grade bands" (A–F)
that dim/light up depending on the last prediction.

### `static/js/script.js`

On form submit, it:

1. Prevents the default page reload.
2. Reads the four field values into a plain object.
3. Disables the button and shows "Predicting..." (basic feedback for the
   round trip).
4. `fetch("/predict", { method: "POST", ... })` with a JSON body.
5. On success: writes the grade into `#prediction`, swaps its color class
   (`grade-a` … `grade-f`) to match, updates the label to show Pass/Fail,
   and lights up the matching grade band.
6. On a `400` or network failure: shows the error message inline under the
   form instead of leaving the user guessing.
7. Always re-enables the button afterward, whether it succeeded or not.

### `templates/history.html`

A plain server-rendered table — no JS needed. Jinja loops over `records`
(built server-side in `/history`) and applies the same grade-color CSS
classes as the predictor page, so a "B" always looks the same color
wherever it appears. Shows a friendly empty-state message if no
predictions have been made yet.

### `static/css/style.css`

One shared stylesheet for both pages, built around a small design system:

- **Palette** — an "ink" navy for headers/text, a warm paper background,
  a ledger green for primary actions, and five grade colors (green
  through red) used consistently for grade bands, the big prediction
  letter, and the history table's grade column.
- **Type** — Fraunces (serif) for headings, IBM Plex Sans for labels and
  body copy, IBM Plex Mono for anything numeric (input values, the big
  grade letter, the accuracy figure) — the idea being that "data" reads
  as data, distinct from prose.
- **Shape** — the input card has a cut notch (like a filed index card);
  the result card reads like a stamped grade. Deliberately different
  shapes so the two cards don't read as identical, interchangeable boxes.

---

## 7. How to modify things later

- **Add a feature** (e.g. "extracurriculars"): add it to `FEATURES` logic
  by adding a column in `generate_dataset.py`, add it to the weighted
  composite, retrain with `train_model.py` (it reads `FEATURES` from the
  dataframe columns you give it — you'd add the new column name to the
  `FEATURES` list at the top of the script), then add a matching
  `<input>` in `index.html` and a line in `script.js`'s `payload` object.
- **Swap in real data**: replace `data/student_data.csv` (same column
  names), rerun `python model/train_model.py`. `app.py` doesn't need any
  changes — it reads whatever `model.pkl` says.
- **Try a different model**: add another `train_and_evaluate(...)` call in
  `train_model.py`'s `results` list; the "pick the best" step already
  works generically off `max(results, key=...)`.
- **Reset prediction history**: delete `prediction_history.csv` — it'll
  be recreated empty on the next prediction or `/history` visit.

---

## 8. Known limitations

- The dataset is synthetic, so accuracy numbers describe how well the
  model recovers a known-but-noisy formula, not real-world performance.
  If real data is substituted, expect the accuracy to change (up or down)
  depending on how noisy real grades actually are relative to these
  features.
- Grades A and F are rare in the generated data (~2% and ~1% of students),
  so the model has very little to learn from for those classes. The
  classification report in `model/metrics.txt` shows this directly (low
  recall on A and F compared to B/C/D). More balanced real-world data, or
  oversampling the rare classes, would improve this.
- There's no authentication or rate-limiting on `/predict` — fine for a
  local/classroom project, but would need hardening before any public
  deployment.