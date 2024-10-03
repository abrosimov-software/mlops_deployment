from src.data.data_preprocessing import preprocess_data
import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data extraction configuration
with open("/app/config/data_management.yaml", 'r') as file:
    data_management_cfg = yaml.safe_load(file)

df = pd.read_csv("/app/data/raw/data.csv")

X_df = df.drop(columns=data_management_cfg["data_description"]["targets"])
y_df = df[data_management_cfg["data_description"]["targets"]]

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

X_train, X_preprocessor = preprocess_data(X_train, data_management_cfg["data_preprocessing"]["features_transformations"])
X_test, _ = preprocess_data(X_test, data_management_cfg["data_preprocessing"]["features_transformations"], preprocessor=X_preprocessor)

X_train.to_csv("/app/data/preprocessed/X_train.csv", index=False)
X_test.to_csv("/app/data/preprocessed/X_test.csv", index=False)

y_train, y_preprocessor = preprocess_data(y_train, data_management_cfg["data_preprocessing"]["targets_transformations"])
y_test, _ = preprocess_data(y_test, data_management_cfg["data_preprocessing"]["targets_transformations"], preprocessor=y_preprocessor)

y_train.to_csv("/app/data/preprocessed/y_train.csv", index=False)
y_test.to_csv("/app/data/preprocessed/y_test.csv", index=False)




