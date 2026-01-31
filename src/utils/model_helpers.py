import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tensorflow import keras
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import optuna
from optuna.samplers import TPESampler
import os
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple


def train_lightgbm_model(
    X: pd.DataFrame, y: pd.Series, params: dict = None, metric: str = "binary_logloss"
) -> lgb.LGBMClassifier:
    """Trains a LightGBM model with given data and parameters.

    ----------
    Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        params (dict, optional): LightGBM parameters. Defaults to None.

    Returns:
        lgb.LGBMClassifier: Trained LightGBM model.
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": metric,
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "n_estimators": 100,
            "random_state": 42,
        }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    return model


def optimize_lightgbm_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    parameter_space: dict = None,
    n_trials: int = 50,
    metric: str = "binary_logloss",
) -> dict:
    """Optimizes LightGBM hyperparameters using Optuna.

    ----------
    Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        n_trials (int, optional): Number of optimization trials. Defaults to 50.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    if parameter_space is None:
        parameter_space = {}

    def objective(trial):

        model = lgb.LGBMClassifier(**parameter_space(trial))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)

        return scores.mean()

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def evaluate_model(
    model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "binary_logloss",
) -> dict:
    """Evaluates the model on validation data and returns performance metrics.

    ----------
    Parameters:
        model (lgb.LGBMClassifier): Trained LightGBM model.
        X_val (pd.DataFrame): Validation feature data.
        y_val (pd.Series): Validation target labels.
        metric (str, optional): Metric to use for evaluation. Defaults to "binary_logloss".
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)
    f1 = f1_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    return {
        "auc": auc,
        "f1_score": f1,
        "classification_report": report,
    }


def model_traininng_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameter_space: dict,
    n_trials: int = 50,
    metric: str = "binary_logloss",
) -> Tuple[lgb.LGBMClassifier, dict]:
    """Complete model training pipeline including hyperparameter optimization and evaluation.

    ----------
    Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target labels.
        X_val (pd.DataFrame): Validation feature data.
        y_val (pd.Series): Validation target labels.
        n_trials (int, optional): Number of optimization trials. Defaults to 50.

    Returns:
        Tuple[lgb.LGBMClassifier, dict]: Trained model and evaluation metrics.
    """
    best_params = optimize_lightgbm_hyperparameters(
        X_train, y_train, parameter_space, n_trials, metric
    )

    model = train_lightgbm_model(X_train, y_train, best_params, metric)

    evaluation_metrics = evaluate_model(model, X_val, y_val)

    return model, evaluation_metrics


def visualize_feature_importance(
    model: lgb.LGBMClassifier, feature_names: list, top_n: int = 20
) -> None:
    """Visualizes the top N feature importances of the model.

    ----------
    Parameters:
        model (lgb.LGBMClassifier): Trained LightGBM model.
        feature_names (list): List of feature names.
        top_n (int, optional): Number of top features to display. Defaults to 20.
    """

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.show()


def save_model(model: lgb.LGBMClassifier, model_path: str) -> None:
    """Saves the trained model to the specified path.

    ----------
    Parameters:
        model (lgb.LGBMClassifier): Trained LightGBM model.
        model_path (str): Path to save the model.
    """
    joblib.dump(model, model_path)
