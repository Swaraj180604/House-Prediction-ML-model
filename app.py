import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 15px;
            background-color: #ffffff;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Title Section
# ---------------------------
st.title("🏠 House Price Prediction System")
st.markdown("### Predict property prices using Machine Learning")
st.write("---")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("🏡 Enter Property Details")

area = st.sidebar.number_input("Area (sq ft)", 100, 10000, 1000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
stories = st.sidebar.slider("Stories", 1, 5, 1)
parking = st.sidebar.slider("Parking", 0, 5, 1)

# ---------------------------
# Main Layout Columns
# ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Property Overview")
    
    st.metric("Area (sq ft)", area)
    st.metric("Bedrooms", bedrooms)
    st.metric("Bathrooms", bathrooms)
    st.metric("Stories", stories)
    st.metric("Parking", parking)

with col2:
    st.subheader("🧮 Prediction")

    if st.button("Predict Price"):
        input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
        prediction = model.predict(input_data)

        st.markdown(f"""
            <div class="prediction-box">
                <h2>💰 Estimated Price</h2>
                <h1>₹ {prediction[0]:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Feature Importance (If Available)
# ---------------------------
if hasattr(model, "feature_importances_"):
    st.write("---")
    st.subheader("📈 Feature Importance")

    features = ["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"]
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    st.pyplot(fig)

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit | Machine Learning Project</center>",
    unsafe_allow_html=True
)
