import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from src.baseline import load_features_target, split_and_scale
from src.regularization import fit_models
from src.coefficient_paths import highlight
n_bootstrap = 100
def bootstrap_coefs(model_class, alpha, X, y):
    rng = np.random.default_rng(42)
    y = np.asarray(y)
    n = len(X)
    coefs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        model = model_class(alpha = alpha, max_iter = 50000).fit(X[idx], y[idx])
        coefs.append(model.coef_)
    return np.array(coefs)
def selection_frequency(coefs):
    return (coefs != 0).mean(axis = 0)
if __name__ == "__main__":
    X, y = load_features_target("dataset/processed/communities_clean.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler = split_and_scale(X, y)
    models = fit_models(X_train_scaled, y_train)
    lasso_boot = bootstrap_coefs(Lasso, models["Lasso"].alpha_, X_train_scaled, y_train)
    ridge_boot = bootstrap_coefs(Ridge, models["Ridge"].alpha_, X_train_scaled, y_train)
    freq = selection_frequency(lasso_boot)
    print(f"Always kept (>= 95%): {(freq >= 0.95).sum()}")
    print(f"Unstable (20% - 80%): {((freq > 0.2) & (freq < 0.8)).sum()}")
    for name in highlight:
        idx = list(feature_names).index(name)
        print(f"Name: {name}")
        print(f"Frequency: {freq[idx]}")
        print(f"Lasso Bootstrap Standard Deviation: {lasso_boot[:, idx].std()}")
        print(f"Ridge Bootstrap Standard Deviation: {ridge_boot[:, idx].std()}")
    freq_series = pd.Series(freq, index = feature_names).sort_values()
    fig, ax = plt.subplots(figsize = (8, 14))
    freq_series.plot.barh(ax = ax)
    ax.axvline(0.95, linestyle = "--", color = "green", label = "always kept (>= 95%)")
    ax.axvline(0.2, linestyle = "--", color = "red", label = "unstable band starts")
    ax.tick_params(axis = "y", labelsize = 6)
    ax.set_xlabel("Selection Frequency across 100 bootstrap resamples")
    ax.set_title("Lasso feature selection stability")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/lasso_selection_frequency.png", dpi = 150)
    plt.show()