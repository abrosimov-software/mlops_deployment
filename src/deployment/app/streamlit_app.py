import streamlit as st
import requests
import yaml
import os

# Load the YAML config
config_path = os.path.join("/app", "config", "data_management.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Get the expected features from the config
features = config["data_description"]["features"]
feature_expectations = config["data_description"]["feature_expectations"]

# Set FastAPI URL from environment variable
fastapi_url = os.getenv("FASTAPI_URL", "http://api:8000")

st.title("ML Model Prediction Interface")

# Create input fields dynamically based on the features
input_data = {}

for feature in features:
    expectation = feature_expectations.get(feature, {})

    if expectation["type"] == "continuous":
        # Set default value as the mean if provided, else 0.0
        default_value = expectation.get("mean", 0.0)
        min_value = expectation.get("min", None)
        max_value = expectation.get("max", None)
        input_data[feature] = st.number_input(
            f"Enter value for {feature}",
            value=float(default_value),
            min_value=float(min_value),
            max_value=float(max_value)
        )

    elif expectation["type"] == "categorical":
        # Set default value as most_frequent if provided, else first category
        categories = expectation.get("categories", [])
        default_value = expectation.get("most_frequent", categories[0] if categories else "")
        input_data[feature] = st.selectbox(
            f"Select value for {feature}",
            options=categories,
            index=categories.index(default_value) if default_value in categories else 0
        )

# When the user clicks the predict button
if st.button("Predict"):
    # Prepare data payload
    payload = {"features": input_data}
    
    # Send a request to FastAPI
    try:
        response = requests.post(f"{fastapi_url}/predict", json=payload)
        response.raise_for_status()  # Check for errors

        # Get the prediction result
        prediction = response.json().get("prediction", [])
        st.success(f"Prediction: {prediction}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
