import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tensorflow import keras
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import numpy as np
import optuna
from optuna.samplers import TPESampler
import os
import joblib
import json

from typing import Tuple


def optimize_lgm_hyper_parameters(
    space: dict,
    n_trials: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metric: str = "auc",
    cv_folds: int = 5,
    seed: int = 42,
):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int(
                space["num_leaves"]["min"], space["num_leaves"]["max"]
            ),
            "max_depth": trial.suggest_int(
                space["max_depth"]["min"], space["max_depth"]["max"]
            ),
            "learning_rate": trial.suggest_float(
                space["learning_rate"]["min"], space["learning_rate"]["max"]
            ),
            "n_estimators": trial.suggest_int(
                space["n_estimators"]["min"], space["n_estimators"]["max"]
            ),
            "min_split_gain": trial.suggest_float(
                space["min_split_gain"]["min"], space["min_split_gain"]["max"]
            ),
            "min_child_samples": trial.suggest_int(
                space["min_child_samples"]["min"], space["min_child_samples"]["max"]
            ),
            "subsample_freq": trial.suggest_int(
                space["subsample_freq"]["min"], space["subsample_freq"]["max"]
            ),
            "subsample": trial.suggest_float(
                space["subsample"]["min"], space["subsample"]["max"]
            ),
            "colsample_bytree": trial.suggest_float(
                space["colsample_bytree"]["min"], space["colsample_bytree"]["max"]
            ),
            "lambda_l1": trial.suggest_float(
                space["lambda_l1"]["min"], space["lambda_l1"]["max"]
            ),
            "lambda_l2": trial.suggest_float(
                space["lambda_l2"]["min"], space["lambda_l2"]["max"]
            ),
        }

        model = lgb.LGBMClassifier(
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            min_split_gain=params["min_split_gain"],
            min_child_samples=params["min_child_samples"],
            subsample_freq=params["subsample_freq"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            lambda_l1=params["lambda_l1"],
            lambda_l2=params["lambda_l2"],
            random_state=seed,
            n_jobs=-1,
        )

        scoring = "roc_auc" if metric == "auc" else "f1"
        X = X_train.drop(columns=["id"]) if "id" in X_train.columns else X_train
        cv_scores = cross_val_score(
            model, X, y_train, cv=skf, scoring=scoring, n_jobs=-1
        )

        mean_score = cv_scores.mean()
        print(
            f"Params: {params}, CV {metric}: {mean_score:.4f} (+/- {cv_scores.std():.4f})"
        )
        return mean_score  # optuna maximizes by default

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    return best_params, study


def train_lgbm_model(
    space: dict,
    n_trials: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_final: pd.DataFrame,
    y_final: pd.Series,
    metric: str = "auc",
    cv_folds: int = 5,
    seed: int = 42,
) -> Tuple[lgb.LGBMClassifier, lgb.LGBMClassifier]:
    best_params, trial = optimize_lgm_hyper_parameters(
        space=space,
        n_trials=n_trials,
        X_train=X_train,
        y_train=y_train,
        metric=metric,
        cv_folds=cv_folds,
    )

    def build_from_params(p):
        return lgb.LGBMClassifier(
            num_leaves=int(p["num_leaves"]),
            max_depth=int(p["max_depth"]),
            learning_rate=float(p["learning_rate"]),
            n_estimators=int(p["n_estimators"]),
            min_split_gain=float(p["min_split_gain"]),
            min_child_samples=int(p["min_child_samples"]),
            subsample_freq=int(p["subsample_freq"]),
            subsample=float(p["subsample"]),
            colsample_bytree=float(p["colsample_bytree"]),
            lambda_l1=float(p["lambda_l1"]),
            lambda_l2=float(p["lambda_l2"]),
            random_state=seed,
            n_jobs=-1,
        )

    model = build_from_params(best_params)

    Xtr = X_train.drop(columns=["id"]) if "id" in X_train.columns else X_train
    Xval = (
        X_validation.drop(columns=["id"])
        if "id" in X_validation.columns
        else X_validation
    )

    model.fit(
        Xtr,
        y_train,
        eval_set=[(Xval, y_validation)],
        eval_metric=metric,
    )

    predictions = model.predict(Xval)
    report = classification_report(y_validation, predictions)
    print(report)

    # ensure reports and models directories
    os.makedirs("../reports", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    report_save_path = (
        f"../reports/classification_report_{len(os.listdir('../reports/')) + 1}.json"
    )
    report_dict = classification_report(y_validation, predictions, output_dict=True)
    with open(report_save_path, "w") as fh:
        json.dump(report_dict, fh, indent=2)

    final_model = build_from_params(best_params)
    Xfinal = X_final.drop(columns=["id"]) if "id" in X_final.columns else X_final
    final_model.fit(
        Xfinal,
        y_final,
        eval_metric=metric,
    )

    model_path = f"../models/lgbm_final_{len(os.listdir('../models')) + 1}.joblib"
    joblib.dump(final_model, model_path)

    return model, final_model, best_params, trial


def train_tensorflow_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.Dense(
                64, activation="relu", input_shape=(X_train.shape[1] - 1,)
            ),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["AUC", "Accuracy"]
    )
    model.fit(
        X_train.drop(columns=["id"]),
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )
    return model
