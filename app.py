from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("models/random_forest_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = []

    for i in range(1, 11):
        features.append(float(request.form[f"f{i}"]))

    prediction = model.predict([features])

    if prediction[0] == 1:
        result = "Fraud Transaction"
    else:
        result = "Legitimate Transaction"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)