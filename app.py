import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")  # Remove if not used

st.title("🏠 House Price Prediction App")

st.write("Enter House Details:")

# Input fields
area = st.number_input("Area (sq ft)")
bedrooms = st.number_input("Number of Bedrooms")
bathrooms = st.number_input("Number of Bathrooms")
stories = st.number_input("Number of Stories")
parking = st.number_input("Parking Spaces")

if st.button("Predict Price"):

    features = np.array([[area, bedrooms, bathrooms, stories, parking]])

    # Apply scaling if used
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
