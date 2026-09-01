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
grades — the current Random Forest model gets about 69% test accuracy. If
you have a real dataset, replace `data/student_data.csv` with columns
`study_hours, attendance, assignment_score, previous_score, grade` and rerun
`model/train_model.py`.