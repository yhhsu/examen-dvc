import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze()

model = joblib.load("models/trained_model.pkl")

numeric_cols = X_test.select_dtypes(include=['float64', 'int64']).columns
X_test = X_test[numeric_cols]

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
metrics = {"MSE": mse, "R2": r2}

pd.DataFrame(predictions, columns=["predictions"]).to_csv("data/processed_data/predictions.csv", index=False)

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)
