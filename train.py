import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



print("Loading ds...")
california = fetch_california_housing()
X = california.data
y = california.target

#split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

experiments = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01}
]


mlflow.set_experiment("House_Price_Prediction")

# 3
for exp in experiments:
    print(f"Training {exp['run_name']}...")
    
    # starting MLflow run
    with mlflow.start_run(run_name=exp["run_name"]):
        
        # Initialize and train model
        model = GradientBoostingRegressor(
            max_depth=exp["max_depth"], 
            learning_rate=exp["learning_rate"], 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Calculate validation metrics
        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        # Log Parameters
        mlflow.log_param("max_depth", exp["max_depth"])
        mlflow.log_param("learning_rate", exp["learning_rate"])
        
        # Log Metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        
        # Log the trained model as an artifact
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        
        print(f"{exp['run_name']} completed. RMSE: {rmse:.4f}")

print("All experiments finished! Run 'mlflow ui' to view the results.")