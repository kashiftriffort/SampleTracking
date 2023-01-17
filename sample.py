import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Create a LogisticRegression model
lr = LogisticRegression()

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the parameters
    mlflow.log_param("alpha", 0.5)
    mlflow.log_param("beta", 0.3)

    # Fit the model
    lr.fit(iris.data, iris.target)

    # Predict and evaluate the model
    predictions = lr.predict(iris.data)
    accuracy = accuracy_score(iris.target, predictions)

    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(lr, "model")
