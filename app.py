from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# ==========================================
# Load Model & Scaler
# ==========================================

MODEL_PATH = "models/floods.save"
SCALER_PATH = "models/scaler.save"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("=" * 60)
print("Model Loaded Successfully")
print("Model Type:", type(model))
print("Model Features:", model.n_features_in_)
print("Scaler Features:", scaler.n_features_in_)
print("=" * 60)

# ==========================================
# Prediction History
# ==========================================

DATA_FOLDER = "data"
HISTORY_FILE = os.path.join(DATA_FOLDER, "prediction_history.csv")

os.makedirs(DATA_FOLDER, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    history = pd.DataFrame(columns=[
        "Date",
        "Prediction",
        "Confidence"
    ])
    history.to_csv(HISTORY_FILE, index=False)

# ==========================================
# Home
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# About
# ==========================================

@app.route("/about")
def about():
    return render_template("about.html")

# ==========================================
# Predict
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        MonsoonIntensity = float(request.form["MonsoonIntensity"])
        DrainageSystems = float(request.form["DrainageSystems"])
        RiverManagement = float(request.form["RiverManagement"])
        Deforestation = float(request.form["Deforestation"])
        Urbanization = float(request.form["Urbanization"])

        input_data = pd.DataFrame([{
            "MonsoonIntensity": MonsoonIntensity,
            "DrainageSystems": DrainageSystems,
            "RiverManagement": RiverManagement,
            "Deforestation": Deforestation,
            "Urbanization": Urbanization
        }])

        print("\n" + "=" * 60)
        print("Input Data")
        print(input_data)

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        print("Prediction :", prediction)
        print("Probability:", probability)
        print("=" * 60)

        confidence = round(float(np.max(probability) * 100), 2)

        if confidence >= 90:
            confidence_level = "Very High"
            confidence_color = "success"

        elif confidence >= 70:
            confidence_level = "High"
            confidence_color = "primary"

        elif confidence >= 50:
            confidence_level = "Moderate"
            confidence_color = "warning"

        else:
            confidence_level = "Low"
            confidence_color = "danger"

        if prediction == 1:

            prediction_text = "⚠ HIGH FLOOD RISK"
            risk = "High"
            color = "danger"

            recommendation = [
                "Issue flood alerts immediately.",
                "Prepare evacuation plans.",
                "Keep emergency shelters ready.",
                "Monitor rainfall continuously.",
                "Deploy emergency response teams."
            ]

        else:

            prediction_text = "✅ LOW FLOOD RISK"
            risk = "Low"
            color = "success"

            recommendation = [
                "Flood risk is currently low.",
                "Continue monitoring weather updates.",
                "Maintain drainage systems.",
                "Stay alert during heavy rainfall.",
                "Follow official weather advisories."
            ]

        history = pd.read_csv(HISTORY_FILE)

        history.loc[len(history)] = [
            datetime.now().strftime("%d-%m-%Y %H:%M"),
            prediction_text,
            confidence
        ]

        history.to_csv(HISTORY_FILE, index=False)

        return render_template(
            "result.html",
            prediction=prediction_text,
            confidence=confidence,
            confidence_level=confidence_level,
            confidence_color=confidence_color,
            risk=risk,
            color=color,
            recommendation=recommendation
        )

    except Exception as e:
        return f"<h2>Error</h2><br>{e}"

# ==========================================
# History
# ==========================================

@app.route("/history")
def history():

    history = pd.read_csv(HISTORY_FILE)

    return render_template(
        "history.html",
        tables=[
            history.to_html(
                classes="table table-striped table-bordered table-hover",
                index=False,
                justify="center"
            )
        ],
        titles=history.columns.values
    )

# ==========================================
# Clear History
# ==========================================

@app.route("/clear_history")
def clear_history():

    history = pd.DataFrame(columns=[
        "Date",
        "Prediction",
        "Confidence"
    ])

    history.to_csv(HISTORY_FILE, index=False)

    return """
    <script>
    alert("Prediction history cleared successfully.");
    window.location.href="/history";
    </script>
    """

# ==========================================
# 404
# ==========================================

@app.errorhandler(404)
def page_not_found(error):

    return """
    <center>
    <h1>404</h1>
    <h2>Page Not Found</h2>
    <br>
    <a href="/">Go Back Home</a>
    </center>
    """, 404

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )