from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("alzheimers_small_model.pkl")

# Print model info in terminal (for verification)
print("Loaded Model Type:", type(model))
print("Model Details:", model)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input values
        features = [float(x) for x in request.form.values()]
        input_data = np.array([features])

        # Make prediction
        prediction = model.predict(input_data)

        # Get probability (since RandomForest supports it)
        probability = model.predict_proba(input_data)
        confidence = round(max(probability[0]) * 100, 2)

        # Convert numeric output to text
        result = "Alzheimer's Disease" if prediction[0] == 1 else "Normal"

        return render_template(
            "index.html",
            prediction_text=result,
            confidence_text=f"Confidence: {confidence}%"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error in prediction",
            confidence_text=str(e)
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)