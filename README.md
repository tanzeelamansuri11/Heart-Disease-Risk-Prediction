# ❤️ Early Detection of Heart Disease Risk Using Machine Learning

## Project Overview

This project uses Machine Learning to estimate the likelihood of heart
disease based on patient health attributes. It uses the UCI Heart
Disease (Cleveland) dataset and compares multiple classification models
through an interactive Streamlit application.

## Objectives

-   Predict the likelihood of heart disease using patient health data.
-   Train and compare different classification algorithms.
-   Evaluate model performance using standard metrics.
-   Build an interactive prediction dashboard.

## Dataset

**Source:** [UCI Heart Disease
Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

The dataset contains 13 patient health attributes, including age, sex,
chest pain type, blood pressure, cholesterol, maximum heart rate and ECG
results.

**Target classes:** - `0` -- No heart disease - `1` -- Heart disease
present

The dataset is fetched using the `ucimlrepo` Python library.

## Machine Learning Models

1.  **Logistic Regression** -- estimates the probability of heart
    disease.
2.  **Random Forest** -- combines multiple decision trees.
3.  **AdaBoost** -- combines weak learners to improve classification.
4.  **Voting Classifier** -- combines the predictions of the three
    models using soft voting.

## Methodology

1.  Fetch and load the dataset.
2.  Preprocess data and handle missing values.
3.  Encode categorical features and scale numerical features.
4.  Split the data into training and testing sets (80:20).
5.  Train the four classification models.
6.  Evaluate and compare their performance.
7.  Save trained models and generate visualizations.

## Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC-AUC
-   Cross-validation

The project can also generate confusion matrices, ROC curves and
feature-importance visualizations.

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit, Joblib and
UCI Machine Learning Repository.

## Project Structure

``` text
heart-disease-risk-prediction/
├── src/
│   ├── __init__.py
│   ├── data.py
│   └── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Generated folders such as `data/`, `models/` and `reports/` may be
created when the training script runs.

## Installation and Execution

**1. Clone the repository**

``` bash
git clone https://github.com/your-username/heart-disease-risk-prediction.git
cd heart-disease-risk-prediction
```

**2. Create and activate a virtual environment (Windows)**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**

``` bash
pip install -r requirements.txt
```

**4. Train the models**

``` bash
python -m src.train
```

The training process fetches the dataset if it is not already cached,
trains the models and saves the generated outputs. An internet
connection is required for the initial dataset download.

**5. Run the Streamlit application**

``` bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Future Scope

-   Hyperparameter tuning.
-   Integration of additional healthcare datasets.
-   Explainable AI techniques.
-   Deployment on a cloud platform.

## Disclaimer

This project is developed for educational purposes only. Predictions are
experimental estimates and must not be considered medical diagnoses or a
substitute for professional medical advice.

## Author

**Early Detection of Heart Disease Risk**\
Machine Learning \| Python \| Streamlit
