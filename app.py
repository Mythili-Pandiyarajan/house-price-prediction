import streamlit as st
import pandas as pd
import pickle

with open("house_price_model.pkl", "rb") as f:
    bundle = pickle.load(f)

st.title("🏠 House Price Predictor")

area = st.number_input("Area (sqft)", min_value=100, value=1800)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
floors = st.slider("Floors", 1, 5, 1)
age = st.number_input("Age of House (years)", min_value=0, value=10)
garage = st.selectbox("Garage", [0, 1])

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "Area_sqft": area, "Bedrooms": bedrooms,
        "Bathrooms": bathrooms, "Floors": floors,
        "Age_of_House": age, "Garage": garage
    }])
    scaled = bundle["scaler"].transform(input_df)
    price = bundle["model"].predict(scaled)[0]
    st.success(f"Predicted Price: ₹{price:,.0f}")