import pandas as pd
import joblib
from sklearn.linear_model import Ridge

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze()

best_params = joblib.load("models/best_params.pkl")

numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train = X_train[numeric_cols]

model = Ridge(**best_params)
model.fit(X_train, y_train)

joblib.dump(model, "models/trained_model.pkl")
