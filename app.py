import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or pickle

# Load trained model
model = joblib.load("random_forest_model.pkl")  # Save your model first!

# App title
st.title("✈️ Flight Price Predictor")

# User inputs
airline = st.selectbox("Airline", ["SpiceJet", "Vistara", "AirAsia","Indigo ","GO_FIRST ","AirAsia","SpiceJet" ]) 
source = st.selectbox("Source City", ["Delhi", "Mumbai","Bangalore","Kolkata","Hyderabad","Chennai"])
destination = st.selectbox("Destination City", ["Delhi", "Mumbai","Bangalore","Kolkata","Hyderabad","Chennai"])
departure_time = st.selectbox("Departure Time", ["Early_Morning ","Morning","Afternoon", "Evening", "Night","Late_Night"])
arrival_time = st.selectbox("Arrival Time", ["Early_Morning ","Morning","Afternoon", "Evening", "Night","Late_Night"])
stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
flight_class = st.selectbox("Class", ["Economy", "Business"])
duration = st.slider("Duration (hours)", 1.0, 20.0, 2.5)
days_left = st.slider("Days Left", 1, 49, 10)

# Build feature vector
input_df = pd.DataFrame({
    "airline": [airline],
    "source_city": [source],
    "destination_city": [destination],
    "departure_time": [departure_time],
    "arrival_time": [arrival_time],
    "stops": [stops],
    "class": [flight_class],
    "duration": [duration],
    "days_left": [days_left]
})

# One-hot encode like training
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Flight Price: ₹{prediction:,.0f}")
