import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
target = "ViolentCrimesPerPop"
sensitive_columns = ["racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"]
random_state = 42
def load_features_target(path):
    df = pd.read_csv(path)
    X = df.drop(columns = [target] + sensitive_columns)
    y = df[target]
    return X, y
def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler
def evaluate(model, X, y):
    pred = model.predict(X)
    return r2_score(y, pred), np.sqrt(mean_squared_error(y, pred))
if __name__ == "__main__":
    X, y = load_features_target("dataset/processed/communities_clean.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler = split_and_scale(X, y)
    model = LinearRegression().fit(X_train_scaled, y_train)
    train_r2_score, train_rmse = evaluate(model, X_train_scaled, y_train)
    test_r2_score, test_rmse = evaluate(model, X_test_scaled, y_test)
    print(f"Train R2 Score: {train_r2_score:.4f}, RMSE: {train_rmse:.4f}")
    print(f"Test R2 Score: {test_r2_score:.4f}, RMSE: {test_rmse:.4f}")
    print(np.linalg.cond(X_train_scaled))
    print(X.shape)