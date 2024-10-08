import mlflow
import mlflow.sklearn
import yaml
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from itertools import product

# Load experiment configurations
with open("/app/config/ml_experiments.yaml", 'r') as file:
    experiments = yaml.safe_load(file)

# Load preprocessed data
X_train = pd.read_csv("/app/data/preprocessed/X_train.csv")
y_train = pd.read_csv("/app/data/preprocessed/y_train.csv")
X_test = pd.read_csv("/app/data/preprocessed/X_test.csv")
y_test = pd.read_csv("/app/data/preprocessed/y_test.csv")


# Function to get all hyperparameter combinations
def get_hyperparameter_combinations(hyperparameters):
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations

# Loop through experiments
for experiment in experiments['experiments']:
    model_name = experiment['model']
    metrics = experiment['metrics']
    
    # Get all combinations of hyperparameters
    hyperparam_combinations = get_hyperparameter_combinations(experiment['hyperparameters'])
    
    # Iterate through each hyperparameter combination
    for i, hyperparam_set in enumerate(hyperparam_combinations):
        
        # Start MLflow run
        run_name = f"{experiment['name']}_{model_name}_run_{i+1}"
        with mlflow.start_run(run_name=run_name):
            
            # Initialize model
            if model_name == 'HistGradientBoostingClassifier':
                model = HistGradientBoostingClassifier(**hyperparam_set)
            elif model_name == 'MLPClassifier':
                model = MLPClassifier(**hyperparam_set)

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log metrics
            if 'accuracy' in metrics:
                accuracy = accuracy_score(y_test, y_pred)
                mlflow.log_metric('accuracy', accuracy)

            # Log hyperparameters and model
            mlflow.log_params(hyperparam_set)
            mlflow.sklearn.log_model(model, model_name)

print("Experiments completed!")
