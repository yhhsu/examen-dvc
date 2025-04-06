import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze()

numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train = X_train[numeric_cols]

model = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
joblib.dump(best_params, "models/best_params.pkl")
