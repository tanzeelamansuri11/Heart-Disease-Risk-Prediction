# Early Detection of Heart Disease Risk

Compares **Logistic Regression, Random Forest, AdaBoost and a soft Voting Classifier** on the
UCI Heart Disease (Cleveland) dataset and serves predictions through **Streamlit**.

## Run in VS Code
1. Open this folder: File > Open Folder > `heart-risk`.
2. Terminal > New Terminal, then create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. In VS Code: Ctrl+Shift+P > "Python: Select Interpreter" > pick `.venv`.
5. Train: `python -m src.train` (downloads the data automatically, needs internet once).
6. App: `streamlit run app.py` (opens http://localhost:8501).

Offline? Put a `heart.csv` in `data/` with columns
`age,trestbps,chol,thalach,oldpeak,ca,sex,cp,fbs,restecg,exang,slope,thal,target` (UCI coding, target 0/1).
Without internet or a CSV, synthetic demo data is used (clearly flagged).
