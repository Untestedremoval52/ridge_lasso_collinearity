import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from src.baseline import load_features_target, split_and_scale, evaluate
ridge_alphas = np.logspace(-2, 4, 50)
lasso_alphas = np.logspace(-5, -1, 50)
l1_ratios = [0.1, 0.5, 0.9]
def fit_models(X_train, y_train):
    ridge = RidgeCV(alphas = ridge_alphas, cv = 5)
    lasso = LassoCV(alphas = lasso_alphas, cv = 5, random_state = 42, max_iter = 50000)
    elastic = ElasticNetCV(l1_ratio = l1_ratios, alphas = lasso_alphas, cv = 5, random_state = 42, max_iter = 50000)
    models = {"Ridge" : ridge, "Lasso" : lasso, "ElasticNet" : elastic}
    for model in models.values():
        model.fit(X_train, y_train)
    return models
def count_nonzero(model):
    return np.count_nonzero(model.coef_)
if __name__ == "__main__":
    X, y = load_features_target("dataset/processed/communities_clean.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler = split_and_scale(X, y)
    models = fit_models(X_train_scaled, y_train)
    for name, model in models.items():
        train_r2_score, train_rmse = evaluate(model, X_train_scaled, y_train)
        test_r2_score, test_rmse = evaluate(model, X_test_scaled, y_test)
        print(f"Name: {name}, Alpha: {model.alpha_}, Train R2 Score: {train_r2_score}, Test R2 Score: {test_r2_score}, Test RMSE: {test_rmse}, Count of non-zero: {count_nonzero(model)}")
    print(models["ElasticNet"].l1_ratio_)