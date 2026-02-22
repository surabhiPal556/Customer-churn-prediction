import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Churn Prediction", layout="centered")

# Title
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    Customer Churn Prediction
    </h1>
""", unsafe_allow_html=True)

st.markdown("### Enter Customer Details")

# Load model
model = tf.keras.models.load_model("model.keras")

# Load encoders
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# -------- INPUTS (ALL BELOW HEADING) -------- #

credit_score = st.number_input("Credit Score", value=0)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

gender = st.selectbox("Gender", ["Male", "Female"])

age = st.slider("Age", 18, 100)

tenure = st.number_input("Tenure", value=0)

balance = st.number_input("Balance", value=0.0)

num_products = st.number_input("Number of Products", value=0)

has_cr_card = st.selectbox("Has Credit Card", [0, 1])

is_active_member = st.selectbox("Is Active Member", [0, 1])

estimated_salary = st.number_input("Estimated Salary", value=0.0)

# -------- PREDICTION -------- #

if st.button("Predict"):

    # Encode gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    # Create input array
    input_data = np.array([[credit_score, gender_encoded, age, tenure,
                            balance, num_products, has_cr_card,
                            is_active_member, estimated_salary]])

    # Combine with geo encoding
    input_data = np.concatenate([input_data, geo_encoded], axis=1)

    # Scale data
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction_proba = model.predict(input_data_scaled)[0][0]

    # Result
    st.subheader("Result")

    if prediction_proba > 0.5:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer will stay ✅")

    # -------- PROBABILITY -------- #

    st.subheader("Churn Probability")

    st.progress(float(prediction_proba))
    st.write(f"Probability: {prediction_proba:.2f}")

    # -------- PIE CHART -------- #

    labels = ['Stay', 'Churn']
    sizes = [1 - prediction_proba, prediction_proba]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig)