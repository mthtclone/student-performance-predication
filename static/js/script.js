const form = document.getElementById('predict-form');
const errorEl = document.getElementById('form-error');
const predictionEl = document.getElementById('prediction');
const predictionLabelEl = document.getElementById('prediction-label');
const bandEls = document.querySelectorAll('.band');
const submitButton = form.querySelector('button');

const GRADE_CLASSES = ['grade-a', 'grade-b', 'grade-c', 'grade-d', 'grade-f'];

function setActiveBand(grade) {
    bandEls.forEach((band) => {
        band.classList.toggle('is-active', band.dataset.grade === grade);
    });
}

function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
}

function clearError() {
    errorEl.hidden = true;
    errorEl.textContent = '';
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();

    const payload = {
        study_hours: form.study_hours.value,
        attendance: form.attendance.value,
        assignment_score: form.assignment_score.value,
        previous_score: form.previous_score.value,
    };

    submitButton.disabled = true;
    submitButton.textContent = 'Predicting...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Something went wrong. Please try again.');
            return;
        }

        predictionEl.textContent = data.grade;
        predictionEl.classList.remove(...GRADE_CLASSES);
        predictionEl.classList.add(`grade-${data.grade.toLowerCase()}`);
        predictionLabelEl.textContent = `Predicted grade \u2014 ${data.result}`;
        setActiveBand(data.grade);
    } catch (err) {
        showError("Couldn't reach the server. Please try again.");
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Predict grade';
    }
});
