import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("🌱 Smart Crop Recommendation System")

st.subheader("Enter Soil and Weather Details")

# Inputs
n = st.number_input("Nitrogen")
p = st.number_input("Phosphorus")
k = st.number_input("Potassium")

temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

# Prediction Button
if st.button("Predict Crop"):

    # Convert input into array
    data = np.array([[n, p, k, temp, humidity, ph, rainfall]])

    # Scale input
    scaled_data = scaler.transform(data)

    # Prediction
    prediction = model.predict(scaled_data)

    # Show result
    st.success(f"🌾 Recommended Crop: {prediction[0]}")
