# Diabetes Risk Analysis

A machine learning web application for early detection of diabetes using lifestyle and genetic data. The project trains Logistic Regression and Random Forest models, selects the best performer, and serves predictions through a Streamlit interface.

## Project Structure

```
.
├── app.py
├── data/
│   └── diabetes.csv
├── models/
├── src/
│   ├── predict.py
│   ├── preprocessing.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python src/train_model.py
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dataset

The sample dataset is located at `data/diabetes.csv`. Replace it with your own CSV file that includes the following columns:

- `age`
- `bmi`
- `physical_activity`
- `diet`
- `family_history`
- `blood_pressure`
- `glucose`
- `diabetes` (target: 1 = diabetic, 0 = not diabetic)

## Notes

- The best-performing model pipeline is saved to `models/best_model.joblib`.
- This project is intended for educational use and does not provide medical advice.
