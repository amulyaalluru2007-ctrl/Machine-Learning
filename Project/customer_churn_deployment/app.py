
from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

MODEL_FILE = "customer_churn_elastic_net.pkl"
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
HISTORY_FILE = "prediction_history.csv"

NUMERICAL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

DATA_URLS = [
    "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
]


# --------------------------------------------------
# RISK CLASSIFICATION
# --------------------------------------------------

def get_risk_level(probability):
    if probability < 30:
        return "Low Risk"
    elif probability < 60:
        return "Medium Risk"
    return "High Risk"


# --------------------------------------------------
# TRAIN THE MACHINE LEARNING MODEL
# --------------------------------------------------

def train_model():

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        last_error = None

        for url in DATA_URLS:
            try:
                df = pd.read_csv(url)
                df.to_csv(DATA_FILE, index=False)
                break
            except Exception as error:
                last_error = error
        else:
            raise RuntimeError(
                f"Dataset download failed: {last_error}"
            )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(" ", np.nan),
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    X = df[FEATURES].copy()
    y = df["Churn"].map({"No": 0, "Yes": 1})

    preprocessor = ColumnTransformer([
        (
            "numerical",
            StandardScaler(),
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        )
    ])

    classifier = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)

    return pipeline


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def get_model():

    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    return train_model()


model = get_model()


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

def get_history():

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(
            columns=["Prediction", "Probability", "Risk"]
        )

    history = pd.read_csv(HISTORY_FILE)

    if "Probability" not in history.columns:
        history["Probability"] = 0.0

    history["Probability"] = pd.to_numeric(
        history["Probability"],
        errors="coerce"
    ).fillna(0.0)

    # Always calculate risk from the stored probability.
    # This also corrects older history records.
    history["Risk"] = history["Probability"].apply(
        get_risk_level
    )

    return history


def save_prediction(result, probability, risk):

    record = pd.DataFrame([{
        "Prediction": result,
        "Probability": round(probability, 2),
        "Risk": risk
    }])

    if os.path.exists(HISTORY_FILE):
        history = get_history()
        history = pd.concat(
            [history, record],
            ignore_index=True
        )
    else:
        history = record

    history.to_csv(HISTORY_FILE, index=False)


# --------------------------------------------------
# HOME AND PREDICTION
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probability = None
    risk_level = None
    error = None

    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            data = {
                "SeniorCitizen": int(
                    request.form["SeniorCitizen"]
                ),
                "tenure": float(
                    request.form["tenure"]
                ),
                "MonthlyCharges": float(
                    request.form["MonthlyCharges"]
                ),
                "TotalCharges": float(
                    request.form["TotalCharges"]
                )
            }

            for feature in CATEGORICAL_FEATURES:
                data[feature] = request.form[feature]

            input_data = pd.DataFrame(
                [data],
                columns=FEATURES
            )

            prediction = int(
                model.predict(input_data)[0]
            )

            probability = float(
                model.predict_proba(input_data)[0][1]
            ) * 100

            if prediction == 1:
                result = "Customer Likely to Churn"
            else:
                result = "Customer Not Likely to Churn"

            risk_level = get_risk_level(probability)

            save_prediction(
                result,
                probability,
                risk_level
            )

        except Exception as exception:
            error = str(exception)

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        risk_level=risk_level,
        error=error,
        form_data=form_data
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    history = get_history()
    total_predictions = len(history)

    if total_predictions:
        churn_count = int(
            (
                history["Prediction"]
                == "Customer Likely to Churn"
            ).sum()
        )

        safe_count = total_predictions - churn_count

        average_probability = round(
            history["Probability"].mean(), 2
        )

        churn_rate = round(
            churn_count / total_predictions * 100, 2
        )

        risk_counts = history["Risk"].value_counts()

        low_risk_count = int(
            risk_counts.get("Low Risk", 0)
        )

        medium_risk_count = int(
            risk_counts.get("Medium Risk", 0)
        )

        high_risk_count = int(
            risk_counts.get("High Risk", 0)
        )

    else:
        churn_count = 0
        safe_count = 0
        average_probability = 0
        churn_rate = 0
        low_risk_count = 0
        medium_risk_count = 0
        high_risk_count = 0

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        churn_count=churn_count,
        safe_count=safe_count,
        average_probability=average_probability,
        churn_rate=churn_rate,
        low_risk_count=low_risk_count,
        medium_risk_count=medium_risk_count,
        high_risk_count=high_risk_count
    )


# --------------------------------------------------
# HISTORY PAGE
# --------------------------------------------------

@app.route("/history")
def history_page():

    history = get_history()
    records = history.to_dict(orient="records")

    return render_template(
        "history.html",
        records=records
    )


# --------------------------------------------------
# SETTINGS PAGE
# --------------------------------------------------

@app.route("/settings")
def settings():

    return render_template(
        "settings.html",
        model_name="Elastic Net Logistic Regression",
        feature_count=len(FEATURES),
        numerical_count=len(NUMERICAL_FEATURES),
        categorical_count=len(CATEGORICAL_FEATURES)
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "ok",
        "model": "Elastic Net Logistic Regression"
    }


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
