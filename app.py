import streamlit as st
import joblib
import pickle
import numpy as np

# Load model and scaler
model = joblib.load("cellphone_linear_model.pkl")
scaler = joblib.load("cellphone_scaler.pkl")

st.title("📱 Mobile Price Prediction App")

st.write("Enter mobile specifications below:")

# User Inputs
weight = st.number_input("Weight (grams)", min_value=0.0)
resoloution = st.number_input("Resolution", min_value=0.0)
ppi = st.number_input("PPI", min_value=0.0)
cpu_core = st.number_input("CPU Core", min_value=1)
cpu_freq = st.number_input("CPU Frequency", min_value=0.0)
internal_mem = st.number_input("Internal Memory (GB)", min_value=0.0)
ram = st.number_input("RAM (GB)", min_value=0.0)
rear_cam = st.number_input("Rear Camera (MP)", min_value=0.0)
front_cam = st.number_input("Front Camera (MP)", min_value=0.0)
battery = st.number_input("Battery (mAh)", min_value=0.0)
thickness = st.number_input("Thickness (mm)", min_value=0.0)

if st.button("Predict Price"):

    input_data = np.array([[weight, resoloution, ppi, cpu_core, cpu_freq,
                            internal_mem, ram, rear_cam, front_cam,
                            battery, thickness]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    st.success(f"Predicted Price: ₹ {prediction[0]:.2f}")