import streamlit as st
import numpy as np
import joblib
import os

st.title("Financial Transaction Fraud Detection")

# Load model
model_path = os.path.join("models", "random_forest_model.pkl")

with open(model_path, "rb") as f:
    model = joblib.load(f)

st.write("Enter transaction details")

time = st.number_input("Time")
amount = st.number_input("Amount")

v1 = st.number_input("V1 - Transaction Frequency Behavior")
v2 = st.number_input("V2 - Device/Login Pattern")
v3 = st.number_input("V3 - Spending Pattern Deviation")
v4 = st.number_input("V4 - Location Pattern")
v5 = st.number_input("V5 - Account Activity Behavior")
v6 = st.number_input("V6 - Payment Method Pattern")
v7 = st.number_input("V7 - Merchant Interaction Behavior")
v8 = st.number_input("V8 - Risk Score Pattern")

if st.button("Predict Fraud"):

    features = np.array([[time, amount, v1, v2, v3, v4, v5, v6, v7, v8]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Legitimate Transaction")