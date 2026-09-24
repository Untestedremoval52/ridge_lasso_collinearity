import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from src.baseline import load_features_target, split_and_scale
from src.regularization import ridge_alphas, lasso_alphas, fit_models
highlight = ["PctKids2Par", "PctFam2Par"]
def compute_path(model_class, alphas, X, y):
    coefs = []
    for alpha in alphas:
        model = model_class(alpha = alpha, max_iter = 50000)
        model.fit(X, y)
        coefs.append(model.coef_)
    return np.array(coefs)
def plot_path(ax, alphas, coefs, feature_names, best_alpha, title):
    ax.plot(alphas, coefs, color = "lightgrey", linewidth = 0.8)
    for name in highlight:
        idx = list(feature_names).index(name)
        ax.plot(alphas, coefs[:, idx], linewidth = 2.5, label = name)
    ax.axvline(best_alpha, linestyle = "--", color = "black", label = "CV - selected alpha")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("coefficient")
    ax.set_title(title)
    ax.legend()
if __name__ == "__main__":
    X, y = load_features_target("dataset/processed/communities_clean.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler = split_and_scale(X, y)
    models = fit_models(X_train_scaled, y_train)
    ridge_coefs = compute_path(Ridge, ridge_alphas, X_train_scaled, y_train)
    lasso_coefs = compute_path(Lasso, lasso_alphas, X_train_scaled, y_train)
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    plot_path(axes[0], ridge_alphas, ridge_coefs, feature_names, models["Ridge"].alpha_, "Ridge (L2)")
    plot_path(axes[1], lasso_alphas, lasso_coefs, feature_names, models["Lasso"].alpha_, "Lasso (L1)")
    plt.tight_layout()
    plt.savefig("reports/figures/coefficient_paths.png", dpi = 150)
    plt.close()