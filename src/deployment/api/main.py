from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import yaml
import os
import pandas as pd

# Load YAML config
config_path = os.path.join("/app", "config", "data_management.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load the preprocessors and model
feature_preprocessor = joblib.load(os.path.join("/app", "models", "preprocessors", "features_preprocessor.pkl"))
target_preprocessor = joblib.load(os.path.join("/app", "models", "preprocessors", "targets_preprocessor.pkl"))
model = joblib.load(os.path.join("/app", "models", "best_model", "best_model.pkl"))

# Define the expected features and targets from config
expected_features = config["data_description"]["features"]
expected_targets = config["data_description"]["targets"]

# Initialize FastAPI
app = FastAPI()

# Define the input format for the API
class InputData(BaseModel):
    features: dict

@app.post("/predict")
async def predict(data: InputData):
    # Validate input
    input_data = data.features
    if not all(feature in input_data for feature in expected_features) or len(input_data) != len(expected_features):
        raise HTTPException(status_code=400, detail="Invalid input features")


    # Extract features in the correct order
    # input_list = [input_data[feature] for feature in expected_features]
    input_data = pd.DataFrame(input_data, index=[0], columns=expected_features)
    
    # Preprocess the input
    processed_input = feature_preprocessor.transform(input_data)
    
    # Predict
    prediction = model.predict(processed_input)
    
    # Post-process the output using target preprocessor
    # final_output = target_preprocessor.inverse_transform(prediction)
    # //TODO - this is not the best way to do it, will fix it later (maybe)
    final_output = target_preprocessor.transformers_[0][1]["transformer"].inverse_transform(prediction.reshape(-1, 1))

    # Return result
    return {"prediction": final_output.tolist()}

