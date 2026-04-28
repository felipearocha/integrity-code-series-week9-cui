"""GBR surrogate for outer-cladding temperature vs source parameters."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "S_mag [m3/m3/s]",
    "theta_crit [-]",
    "i0_ref [A/m2]",
    "E_a [J/mol]",
    "lambda_eff [W/mK]",
    "L_defect [-]",
]


def build_feature_matrix(params):
    keys = ["S_mag", "theta_crit", "i0_ref", "E_a", "lambda_eff", "L_defect"]
    return np.column_stack([params[k] for k in keys])


def train_surrogate(
    params,
    wall_loss,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    test_size=0.15,
):
    X = build_feature_matrix(params)
    y = wall_loss
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    gbr = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
        loss="huber",
    )
    gbr.fit(X_train, y_train)
    y_pred_train = gbr.predict(X_train)
    y_pred_test = gbr.predict(X_test)
    perm = permutation_importance(
        gbr, X_test, y_test, n_repeats=10, random_state=random_state, scoring="r2"
    )
    return {
        "model": gbr,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "feature_importance": perm.importances_mean,
        "feature_names": FEATURE_NAMES,
    }


def predict_surrogate(surr, params):
    X = build_feature_matrix(params)
    return np.clip(surr["model"].predict(X), 0.0, 20.0)
