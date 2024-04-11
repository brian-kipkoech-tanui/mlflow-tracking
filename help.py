import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

exp_id = mlflow.create_experiment("Loan_Prediction_Model")

with mlflow.start_run(run_name="DecisionTree") as run:
    mlflow.set_tag("version", "1.0.0")

mlflow.end_run()

n_estimators=10
criterion='gini'
