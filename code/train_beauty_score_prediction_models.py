import os
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import shap
except Exception:
    shap = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# =========================
# Path configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = PROJECT_ROOT / "inputs" / "all_dataset.csv"
PATIENT_INFO_XLSX = PROJECT_ROOT / "inputs" / "patient_info.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
IMPORTANCE_DIR = OUTPUT_DIR / "feature_importance"
PLOT_DIR = OUTPUT_DIR / "plots"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
ENABLE_HYPERPARAMETER_SEARCH = True
DEFAULT_SEARCH_ITER = 18
SEARCH_N_JOBS = -1
BEAUTIFUL_THRESHOLD = 7.0
TOP_N_FEATURES = 30
SHAP_BACKGROUND_SIZE = 100
SHAP_EVAL_SIZE = 100

LEGACY_SAMPLE_NAME_COLUMN = "\u6837\u672c\u540d"
LEGACY_PERSON_NAME_COLUMN = "\u59d3\u540d"
ID_COLUMNS = ["sample_name", "name", LEGACY_SAMPLE_NAME_COLUMN, LEGACY_PERSON_NAME_COLUMN, "stage", "split", "image", "excel_batch"]
TARGET_COLUMN = "beauty_score"
AUX_CLASS_COLUMN = "is_beautiful"
FEATURE_PREFIX = "measurement__"
SPLIT_COLUMN = "split"
EXCEL_BATCH_COLUMN = "excel_batch"
TEST_BATCHES = ["6", "7"]
TRAIN_DATASET_FILENAME = "fixed_batch_train_dataset.csv"
TEST_DATASET_FILENAME = "fixed_batch_test_dataset.csv"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs():
    for path in [OUTPUT_DIR, MODEL_DIR, PRED_DIR, METRIC_DIR, IMPORTANCE_DIR, PLOT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", str(name))
    return name.strip("_")[:120]


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace(" ", "")


def normalize_batch(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "unlabeled"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return str(value).strip()


def load_patient_batch_map():
    info_df = pd.read_excel(PATIENT_INFO_XLSX, sheet_name=0)
    name_col = info_df.columns[0]
    batch_col = info_df.columns[4]
    batch_df = pd.DataFrame({
        "name_key": info_df[name_col].map(normalize_name),
        EXCEL_BATCH_COLUMN: info_df[batch_col].map(normalize_batch),
    })
    batch_df = batch_df[batch_df["name_key"] != ""].drop_duplicates("name_key", keep="first")
    return batch_df


def load_dataset():
    df = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    feature_columns = [col for col in df.columns if col.startswith(FEATURE_PREFIX)]
    if not feature_columns:
        raise ValueError(f"No measurement feature columns starting with {FEATURE_PREFIX!r} were found in {DATASET_CSV}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Regression target column {TARGET_COLUMN!r} is missing from {DATASET_CSV}")

    df = df.dropna(subset=feature_columns + [TARGET_COLUMN]).reset_index(drop=True)
    if AUX_CLASS_COLUMN not in df.columns:
        df[AUX_CLASS_COLUMN] = (df[TARGET_COLUMN] >= BEAUTIFUL_THRESHOLD).astype(int)

    name_column = next(
        (column for column in ("name", LEGACY_PERSON_NAME_COLUMN, "sample_name", LEGACY_SAMPLE_NAME_COLUMN) if column in df.columns),
        None,
    )
    if name_column is None:
        raise ValueError(f"No subject-name column was found in {DATASET_CSV}")
    batch_df = load_patient_batch_map()
    df["name_key"] = df[name_column].map(normalize_name)
    df = df.merge(batch_df, on="name_key", how="left")
    if SPLIT_COLUMN in df.columns:
        supplementary_mask = df[SPLIT_COLUMN].astype(str).str.startswith("supplementary")
        df.loc[supplementary_mask & df[EXCEL_BATCH_COLUMN].isna(), EXCEL_BATCH_COLUMN] = df.loc[supplementary_mask, SPLIT_COLUMN].astype(str)
    df[EXCEL_BATCH_COLUMN] = df[EXCEL_BATCH_COLUMN].fillna("unmatched")

    test_mask = df[EXCEL_BATCH_COLUMN].astype(str).isin(TEST_BATCHES)
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty:
        raise ValueError(f"No training samples remain in {DATASET_CSV} (excel_batch is not in {TEST_BATCHES})")
    if test_df.empty:
        raise ValueError(f"No holdout samples with a fixed test batch in {DATASET_CSV} (excel_batch in {TEST_BATCHES})")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].astype(float).copy()
    y_test = test_df[TARGET_COLUMN].astype(float).copy()

    train_ids = train_df[[col for col in ID_COLUMNS if col in train_df.columns]].copy()
    test_ids = test_df[[col for col in ID_COLUMNS if col in test_df.columns]].copy()
    return train_df, test_df, train_ids, test_ids, X_train, X_test, y_train, y_test, feature_columns


def build_models():
    alphas = np.logspace(-3, 3, 13)
    model_specs = {
        "LinearRegression": {
            "estimator": LinearRegression(),
            "param_distributions": {},
            "search_note": "No key hyperparameters; fit directly as a linear baseline.",
        },
        "RidgeCV": {
            "estimator": RidgeCV(alphas=alphas),
            "param_distributions": {},
            "search_note": "The estimator uses internal cross-validation to select the L2 regularization strength alpha.",
        },
        "LassoCV": {
            "estimator": LassoCV(
                alphas=np.logspace(-4, 1, 40),
                cv=CV_FOLDS,
                max_iter=50000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_distributions": {},
            "search_note": "The estimator uses internal cross-validation to select the L1 regularization strength alpha.",
        },
        "ElasticNetCV": {
            "estimator": ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                alphas=np.logspace(-3, 1, 20),
                cv=CV_FOLDS,
                max_iter=30000,
                random_state=RANDOM_STATE,
            ),
            "param_distributions": {},
            "search_note": "The estimator uses internal cross-validation to select alpha and l1_ratio jointly.",
        },
        "PLSRegression": {
            "estimator": PLSRegression(),
            "param_distributions": {"n_components": [2, 3, 5, 8, 10, 12, 15, 20, 25, 30]},
            "n_iter": 10,
            "search_note": "The measurement features may be collinear; the search focuses on the number of latent components.",
        },
        "SVR_RBF": {
            "estimator": SVR(kernel="rbf"),
            "param_distributions": {
                "C": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                "epsilon": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
                "gamma": ["scale", "auto", 0.003, 0.01, 0.03, 0.1],
            },
            "search_note": "For standardized small-sample regression, search C, epsilon, and gamma to control smoothness and the tolerance band.",
        },
        "KNeighbors": {
            "estimator": KNeighborsRegressor(metric="minkowski"),
            "param_distributions": {
                "n_neighbors": [3, 5, 7, 9, 12, 15, 20, 30],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "search_note": "Search the number of neighbors, distance weighting, and L1/L2 distance in standardized feature space.",
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            "param_distributions": {
                "n_estimators": [300, 500, 800],
                "max_features": ["sqrt", 0.4, 0.6, 0.8, 1.0],
                "max_depth": [None, 4, 6, 8, 12],
                "min_samples_leaf": [1, 2, 4, 6, 8],
                "min_samples_split": [2, 5, 10],
            },
            "search_note": "Control tree depth, leaf sample size, and feature subsampling to reduce overfitting.",
        },
        "ExtraTrees": {
            "estimator": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
            "param_distributions": {
                "n_estimators": [300, 500, 800],
                "max_features": ["sqrt", 0.4, 0.6, 0.8, 1.0],
                "max_depth": [None, 4, 6, 8, 12],
                "min_samples_leaf": [1, 2, 4, 6],
                "min_samples_split": [2, 5, 10],
            },
            "search_note": "Add further randomization than a random forest and tune complexity to balance variance.",
        },
    }

    if XGBRegressor is not None:
        model_specs["XGBoost"] = {
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "param_distributions": {
                "n_estimators": [300, 500, 800],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_alpha": [0.0, 0.01, 0.05, 0.1],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            },
            "search_note": "Tune learning rate, tree depth, sampling ratios, and regularization for gradient-boosted trees.",
        }
    if LGBMRegressor is not None:
        model_specs["LightGBM"] = {
            "estimator": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, force_col_wise=True),
            "param_distributions": {
                "n_estimators": [300, 500, 800],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
                "num_leaves": [7, 15, 31, 63],
                "max_depth": [-1, 3, 5, 7],
                "min_child_samples": [10, 20, 40, 80],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_alpha": [0.0, 0.01, 0.05, 0.1],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            },
            "n_iter": 12,
            "search_note": "For tree-based regression, focus on leaf count, min_child_samples, sampling, and regularization.",
        }
    if CatBoostRegressor is not None:
        model_specs["CatBoost"] = {
            "estimator": CatBoostRegressor(
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
            "param_distributions": {
                "iterations": [300, 500, 800],
                "depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
                "l2_leaf_reg": [1, 3, 5, 10, 20],
            },
            "search_note": "For small-sample tabular regression, search depth, learning rate, and L2 leaf regularization.",
        }
    return model_specs


def tune_or_fit_model(model_name, spec, X_train_scaled, y_train_scaled):
    estimator = spec["estimator"]
    param_distributions = spec.get("param_distributions", {})
    note = spec.get("search_note", "")

    if not ENABLE_HYPERPARAMETER_SEARCH or not param_distributions:
        estimator.fit(X_train_scaled, y_train_scaled)
        best_params = getattr(estimator, "get_params", lambda: {})()
        return estimator, {
            "model": model_name,
            "search_type": "direct_or_internal_cv",
            "cv_folds": CV_FOLDS if "CV" in model_name else 0,
            "n_candidates": 1,
            "best_cv_neg_rmse_scaled": np.nan,
            "best_cv_rmse_scaled": np.nan,
            "best_params": best_params,
            "note": note,
        }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_iter = min(spec.get("n_iter", DEFAULT_SEARCH_ITER), int(np.prod([len(v) for v in param_distributions.values()])))
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=SEARCH_N_JOBS,
        verbose=1,
        return_train_score=True,
        error_score=np.nan,
    )
    search.fit(X_train_scaled, y_train_scaled)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(METRIC_DIR / f"{model_name}_hyperparameter_search_cv_results.csv", index=False, encoding="utf-8-sig")
    return search.best_estimator_, {
        "model": model_name,
        "search_type": "RandomizedSearchCV",
        "cv_folds": CV_FOLDS,
        "n_candidates": n_iter,
        "best_cv_neg_rmse_scaled": search.best_score_,
        "best_cv_rmse_scaled": -search.best_score_,
        "best_params": search.best_params_,
        "note": note,
    }


def scale_datasets(X_train, X_test, y_train):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(x_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(x_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
    return X_train_scaled, X_test_scaled, y_train_scaled, x_scaler, y_scaler


def inverse_transform_target(y_scaler, y_pred_scaled):
    return y_scaler.inverse_transform(np.asarray(y_pred_scaled).reshape(-1, 1)).ravel()


def tolerance_accuracy(y_true, y_pred, tolerance=0.10):
    denom = np.maximum(np.abs(y_true), 1e-8)
    rel_error = np.abs(y_pred - y_true) / denom
    return float(np.mean(rel_error <= tolerance))


def evaluate_predictions(model_name, y_true, y_pred, split):
    true_class = (np.asarray(y_true) >= BEAUTIFUL_THRESHOLD).astype(int)
    pred_class = (np.asarray(y_pred) >= BEAUTIFUL_THRESHOLD).astype(int)
    return {
        "model": model_name,
        "split": split,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "Acc@0.5score": float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)) <= 0.5)),
        "Acc@1.0score": float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)) <= 1.0)),
        "Acc@5pct": tolerance_accuracy(y_true, y_pred, tolerance=0.05),
        "Acc@10pct": tolerance_accuracy(y_true, y_pred, tolerance=0.10),
        "beautiful_accuracy": accuracy_score(true_class, pred_class),
        "beautiful_precision": precision_score(true_class, pred_class, zero_division=0),
        "beautiful_recall": recall_score(true_class, pred_class, zero_division=0),
        "beautiful_f1": f1_score(true_class, pred_class, zero_division=0),
    }


def save_predictions(model_name, ids_df, y_true, y_pred, split):
    pred_df = ids_df.copy()
    pred_df["true_beauty_score"] = np.asarray(y_true)
    pred_df["pred_beauty_score"] = np.asarray(y_pred)
    pred_df["error_beauty_score"] = pred_df["pred_beauty_score"] - pred_df["true_beauty_score"]
    pred_df["true_is_beautiful"] = (pred_df["true_beauty_score"] >= BEAUTIFUL_THRESHOLD).astype(int)
    pred_df["pred_is_beautiful"] = (pred_df["pred_beauty_score"] >= BEAUTIFUL_THRESHOLD).astype(int)
    pred_df.to_csv(PRED_DIR / f"{model_name}_{split}_predictions.csv", index=False, encoding="utf-8-sig")


def sample_background(X_train, X_test):
    bg = X_train.sample(min(SHAP_BACKGROUND_SIZE, len(X_train)), random_state=RANDOM_STATE)
    ev = X_test.sample(min(SHAP_EVAL_SIZE, len(X_test)), random_state=RANDOM_STATE)
    return bg, ev


def normalize_importance(values, feature_names):
    values = np.asarray(values, dtype=float)
    if values.ndim > 1:
        values = values.reshape(-1)
    if len(values) != len(feature_names):
        values = np.resize(values, len(feature_names))
    return values


def builtin_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return normalize_importance(model.feature_importances_, feature_names), "fallback_feature_importances"
    if hasattr(model, "coef_"):
        return normalize_importance(np.abs(model.coef_), feature_names), "fallback_abs_coefficient"
    return np.zeros(len(feature_names), dtype=float), "fallback_zero_importance"


def permutation_importance_for_model(model, X_eval, y_eval_scaled, feature_names):
    result = permutation_importance(
        model,
        X_eval,
        y_eval_scaled,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return np.maximum(result.importances_mean, 0), "fallback_permutation_importance"


def explain_model(model_name, model, X_train, X_test, y_test_scaled, feature_names):
    X_background, X_eval = sample_background(X_train, X_test)
    method = ""

    try:
        if shap is None:
            raise RuntimeError("shap is not installed")
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_background)
            shap_values = explainer.shap_values(X_eval)
            importance = normalize_importance(np.abs(shap_values).mean(axis=0), feature_names)
            method = "SHAP_LinearExplainer"
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_eval)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            importance = normalize_importance(np.abs(shap_values).mean(axis=0), feature_names)
            method = "SHAP_TreeExplainer"
    except Exception as error:
        importance, method = builtin_importance(model, feature_names)
        if method == "fallback_zero_importance":
            importance, method = permutation_importance_for_model(model, X_eval, y_test_scaled.loc[X_eval.index], feature_names)
        print(f"[Warning] SHAP computation failed for {model_name}; using {method}: {error}")

    importance_df = pd.DataFrame({
        "model": model_name,
        "feature": feature_names,
        "importance": importance,
        "explain_method": method,
    }).sort_values("importance", ascending=False)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    importance_df.to_csv(IMPORTANCE_DIR / f"{model_name}_feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(model_name, importance_df)
    return importance_df


def plot_feature_importance(model_name, importance_df):
    top_df = importance_df.head(TOP_N_FEATURES).iloc[::-1]
    plt.figure(figsize=(10, max(6, 0.28 * len(top_df))))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title(f"{model_name} Top {len(top_df)} Measurement Importance")
    plt.xlabel("Mean |SHAP value| / importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{model_name}_top_feature_importance.png", dpi=200)
    plt.close()


def plot_metrics(metrics_df):
    test_metrics = metrics_df[metrics_df["split"] == "test"].sort_values("R2", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=test_metrics, x="R2", y="model")
    plt.title("Beauty Score Regression Test R2 by Model")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_r2_by_model.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=test_metrics, x="MAE", y="model")
    plt.title("Beauty Score Regression Test MAE by Model")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_mae_by_model.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=test_metrics, x="beautiful_f1", y="model")
    plt.title(f"Auxiliary Beautiful Classification F1 by Model (score >= {BEAUTIFUL_THRESHOLD:g})")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "test_aux_beautiful_f1_by_model.png", dpi=200)
    plt.close()


def plot_best_model_outputs(best_model_name, y_test, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.75, edgecolor="k", linewidth=0.3)
    min_v = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.axhline(BEAUTIFUL_THRESHOLD, color="gray", linestyle=":", linewidth=1)
    plt.axvline(BEAUTIFUL_THRESHOLD, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("True beauty score")
    plt.ylabel("Predicted beauty score")
    plt.title(f"Best Model Test Scatter: {best_model_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{best_model_name}_test_scatter.png", dpi=220)
    plt.close()

    cm = confusion_matrix((np.asarray(y_test) >= BEAUTIFUL_THRESHOLD).astype(int), (np.asarray(y_pred) >= BEAUTIFUL_THRESHOLD).astype(int))
    plt.figure(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["not_beautiful", "beautiful"], yticklabels=["not_beautiful", "beautiful"])
    plt.xlabel("Predicted class from score")
    plt.ylabel("True class from score")
    plt.title(f"Auxiliary Confusion Matrix: {best_model_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{best_model_name}_aux_confusion_matrix.png", dpi=220)
    plt.close()


def format_float(value, digits=4):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def compact_params(row):
    """Show only key hyperparameters to keep Markdown tables readable."""
    search_type = str(row.get("search_type", ""))
    if search_type != "RandomizedSearchCV":
        return "Internal CV / default parameters"

    params_text = str(row.get("best_params", ""))
    params_text = params_text.replace("|", "/").replace("\n", " ")
    params_text = re.sub(r"\s+", " ", params_text).strip()
    return params_text


def top_feature_table(top_n=15):
    path = IMPORTANCE_DIR / "overall_mean_feature_importance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["rank", "feature", "mean_importance", "max_importance"])
    df = pd.read_csv(path, encoding="utf-8-sig").head(top_n).copy()
    return df[["rank", "feature", "mean_importance", "max_importance"]]


def relative_output_path(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR).as_posix()


def write_markdown_report(metrics_df, best_model_name, dataset_info, model_times, search_summary_df):
    if "model_selection_score" in metrics_df.columns:
        test_metrics = metrics_df[metrics_df["split"] == "test"].sort_values("model_selection_score", ascending=True)
        selection_label = dataset_info.get("best_model_selection_policy", "Overall ranking")
    else:
        test_metrics = metrics_df[metrics_df["split"] == "test"].sort_values("R2", ascending=False)
        selection_label = "Test-set R2"
    train_metrics = metrics_df[metrics_df["split"] == "train"]
    metric_columns = ["model", "MAE", "RMSE", "R2", "Acc@0.5score", "Acc@1.0score", "beautiful_accuracy", "beautiful_f1"]
    best_row = test_metrics.iloc[0]
    feature_df = top_feature_table(top_n=15)

    overfit_df = test_metrics[["model", "MAE", "RMSE", "R2"]].merge(
        train_metrics[["model", "R2"]].rename(columns={"R2": "train_R2"}),
        on="model",
        how="left",
    )
    overfit_df["R2_gap_train_minus_test"] = overfit_df["train_R2"] - overfit_df["R2"]

    lines = []
    lines.append("# Beauty Score Regression Model Report\n\n")
    lines.append("> The primary task is to predict the continuous `beauty_score`. `is_beautiful` is an auxiliary binary view derived from a score threshold.\n\n")

    lines.append("## 1. Summary\n\n")
    lines.append("| Item | Result |\n")
    lines.append("|---|---:|\n")
    lines.append(f"| Best model ({selection_label}) | **{best_model_name}** |\n")
    lines.append(f"| Test-set R2 | **{format_float(best_row['R2'])}** |\n")
    lines.append(f"| Test-set MAE | **{format_float(best_row['MAE'])}** |\n")
    lines.append(f"| Test-set RMSE | **{format_float(best_row['RMSE'])}** |\n")
    lines.append(f"| Predictions within 1 score point | **{format_float(best_row['Acc@1.0score'])}** |\n")
    lines.append(f"| Sample count | {dataset_info.get('total_samples_after_dropna', '-')} |\n")
    lines.append(f"| Train / test samples | {dataset_info.get('train_samples', '-')} / {dataset_info.get('test_samples', '-')} |\n")
    lines.append(f"| Number of input features | {dataset_info.get('input_features', '-')} |\n")

    lines.append("\n## 2. Task and Data\n\n")
    lines.append(f"- Source dataset: `{dataset_info.get('source_dataset', DATASET_CSV)}`\n")
    lines.append("- Inputs: 50 scalar `measurement__*` features.\n")
    lines.append("- Target: continuous `beauty_score`.\n")
    lines.append(f"- Auxiliary classification: scores `>= {BEAUTIFUL_THRESHOLD:g}` are labeled as beautiful for evaluation only; no classifier is trained.\n")
    lines.append(f"- Split: fixed holdout by batch from the subject-information workbook; `{EXCEL_BATCH_COLUMN}` in `{', '.join(TEST_BATCHES)}` is the test set.\n")
    lines.append("- Preprocessing: fit `StandardScaler` on training features and target scores; inverse-transform predictions before evaluation.\n")

    lines.append(f"\n## 3. Test-Set Model Ranking ({selection_label})\n\n")
    lines.append("| Rank | Model | MAE↓ | RMSE↓ | R2↑ | Within 0.5 | Within 1.0 | Auxiliary accuracy | Auxiliary F1 |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for rank, (_, row) in enumerate(test_metrics[metric_columns].iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['model']} | {format_float(row['MAE'])} | {format_float(row['RMSE'])} | "
            f"{format_float(row['R2'])} | {format_float(row['Acc@0.5score'])} | {format_float(row['Acc@1.0score'])} | "
            f"{format_float(row['beautiful_accuracy'])} | {format_float(row['beautiful_f1'])} |\n"
        )

    lines.append("\n> Note: An auxiliary classification F1 of zero can occur when the test set has few high-scoring subjects and predictions do not cross the score threshold. This does not change the primary regression metrics.\n")

    lines.append("\n## 4. Generalization (Train R2 vs. Test R2)\n\n")
    lines.append("| Model | Train R2 | Test R2 | R2 gap |\n")
    lines.append("|---|---:|---:|---:|\n")
    for _, row in overfit_df.iterrows():
        lines.append(f"| {row['model']} | {format_float(row['train_R2'])} | {format_float(row['R2'])} | {format_float(row['R2_gap_train_minus_test'])} |\n")

    lines.append("\n## 5. Hyperparameter Search Summary\n\n")
    lines.append("| Model | Search method | Candidates | CV RMSE↓ (standardized space) | Best/key parameters |\n")
    lines.append("|---|---|---:|---:|---|\n")
    for _, row in search_summary_df.iterrows():
        cv_rmse = format_float(row["best_cv_rmse_scaled"]) if "best_cv_rmse_scaled" in row else "-"
        lines.append(
            f"| {row['model']} | {row['search_type']} | {int(row['n_candidates'])} | {cv_rmse} | `{compact_params(row)}` |\n"
        )
    lines.append("\nFull search results are available in `metrics/hyperparameter_search_summary.csv` and `metrics/<model>_hyperparameter_search_cv_results.csv`.\n")

    lines.append("\n## 6. Overall Feature Importance: Top 15\n\n")
    if feature_df.empty:
        lines.append("No overall feature-importance file has been generated.\n")
    else:
        lines.append("| Rank | Feature | Mean importance | Maximum importance |\n")
        lines.append("|---:|---|---:|---:|\n")
        for _, row in feature_df.iterrows():
            lines.append(
                f"| {int(row['rank'])} | `{row['feature']}` | {format_float(row['mean_importance'], 6)} | {format_float(row['max_importance'], 6)} |\n"
            )

    lines.append("\n## 7. Key Figures\n\n")
    plot_files = [
        PLOT_DIR / "test_r2_by_model.png",
        PLOT_DIR / "test_mae_by_model.png",
        PLOT_DIR / f"{best_model_name}_test_scatter.png",
        PLOT_DIR / f"{best_model_name}_top_feature_importance.png",
    ]
    for path in plot_files:
        if path.exists():
            rel_path = relative_output_path(path)
            title = path.stem.replace("_", " ")
            lines.append(f"### {title}\n\n![{title}]({rel_path})\n\n")

    if model_times:
        lines.append("\n## 8. Training Time\n\n")
        lines.append("| Model | Seconds |\n")
        lines.append("|---|---:|\n")
        for model_name, seconds in model_times.items():
            lines.append(f"| {model_name} | {seconds:.2f} |\n")

    lines.append("\n## 9. Output Files\n\n")
    lines.append("| Type | File |\n")
    lines.append("|---|---|\n")
    lines.append("| Metrics summary | `metrics/metrics_summary.csv` |\n")
    lines.append("| Hyperparameter summary | `metrics/hyperparameter_search_summary.csv` |\n")
    lines.append("| Predictions | `predictions/*_test_predictions.csv` |\n")
    lines.append("| Feature importance | `feature_importance/*_feature_importance.csv` |\n")
    lines.append("| Overall feature importance | `feature_importance/overall_mean_feature_importance.csv` |\n")
    lines.append("| Figures | `plots/*.png` |\n")
    (OUTPUT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")


def regenerate_report_from_outputs():
    metrics_path = METRIC_DIR / "metrics_summary.csv"
    search_path = METRIC_DIR / "hyperparameter_search_summary.csv"
    train_path = OUTPUT_DIR / TRAIN_DATASET_FILENAME
    test_path = OUTPUT_DIR / TEST_DATASET_FILENAME
    if not metrics_path.exists() or not search_path.exists():
        raise FileNotFoundError("Trained outputs are missing. Run the complete training script first.")

    metrics_df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    search_summary_df = pd.read_csv(search_path, encoding="utf-8-sig")
    best_row = metrics_df[metrics_df["split"] == "test"].sort_values("R2", ascending=False).iloc[0]
    train_samples = len(pd.read_csv(train_path, encoding="utf-8-sig")) if train_path.exists() else "-"
    test_samples = len(pd.read_csv(test_path, encoding="utf-8-sig")) if test_path.exists() else "-"
    dataset_info = {
        "source_dataset": str(DATASET_CSV),
        "total_samples_after_dropna": train_samples + test_samples if isinstance(train_samples, int) and isinstance(test_samples, int) else "-",
        "train_samples": train_samples,
        "test_samples": test_samples,
        "split_mode": f"fixed {EXCEL_BATCH_COLUMN} in {TEST_BATCHES}",
        "input_features": 50,
        "best_model_by_test_R2": best_row["model"],
        "best_model_test_R2": f"{best_row['R2']:.4f}",
        "best_model_test_MAE": f"{best_row['MAE']:.4f}",
        "best_model_test_RMSE": f"{best_row['RMSE']:.4f}",
    }
    write_markdown_report(metrics_df, best_row["model"], dataset_info, {}, search_summary_df)
    print(f"[Report] Regenerated: {OUTPUT_DIR / 'report.md'}")


def main():
    ensure_dirs()
    train_df, test_df, train_ids, test_ids, X_train, X_test, y_train, y_test, feature_names = load_dataset()
    X_train_scaled, X_test_scaled, y_train_scaled, x_scaler, y_scaler = scale_datasets(X_train, X_test, y_train)
    y_test_scaled = pd.Series(y_scaler.transform(y_test.to_numpy().reshape(-1, 1)).ravel(), index=y_test.index)

    print("=" * 70)
    print("Beauty score regression from measurements")
    print(f"Dataset: {DATASET_CSV}")
    print(f"Fixed batch split: train {len(X_train)}, test {len(X_test)}, test {EXCEL_BATCH_COLUMN} in {TEST_BATCHES}")
    print(f"Input measurement features: {len(feature_names)}")
    print(f"Main target: {TARGET_COLUMN}; auxiliary class threshold: score >= {BEAUTIFUL_THRESHOLD:g}")
    print("=" * 70)

    train_df.to_csv(OUTPUT_DIR / TRAIN_DATASET_FILENAME, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / TEST_DATASET_FILENAME, index=False, encoding="utf-8-sig")
    joblib.dump(x_scaler, MODEL_DIR / "input_feature_standard_scaler.joblib")
    joblib.dump(y_scaler, MODEL_DIR / "target_score_standard_scaler.joblib")

    model_specs = build_models()
    metric_records = []
    importance_records = []
    model_times = {}
    test_predictions = {}
    search_records = []

    for model_name, spec in model_specs.items():
        print(f"\n[Optimize/Train] {model_name}")
        start_time = time.time()
        model, search_record = tune_or_fit_model(model_name, spec, X_train_scaled, y_train_scaled)
        elapsed = time.time() - start_time
        model_times[model_name] = elapsed
        search_records.append(search_record)
        if pd.isna(search_record["best_cv_rmse_scaled"]):
            print(f"[Done] {model_name} fitted in {elapsed:.2f}s ({search_record['search_type']})")
        else:
            print(f"[Done] {model_name} search+fit time: {elapsed:.2f}s, best CV RMSE(scaled): {search_record['best_cv_rmse_scaled']:.4f}")
            print(f"[Best Params] {search_record['best_params']}")

        train_pred = inverse_transform_target(y_scaler, model.predict(X_train_scaled))
        test_pred = inverse_transform_target(y_scaler, model.predict(X_test_scaled))
        test_predictions[model_name] = test_pred

        metric_records.append(evaluate_predictions(model_name, y_train.values, train_pred, "train"))
        metric_records.append(evaluate_predictions(model_name, y_test.values, test_pred, "test"))

        save_predictions(model_name, train_ids, y_train.values, train_pred, "train")
        save_predictions(model_name, test_ids, y_test.values, test_pred, "test")
        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")

        print(f"[Explain] {model_name} feature importance")
        importance_records.append(explain_model(model_name, model, X_train_scaled, X_test_scaled, y_test_scaled, feature_names))

    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(METRIC_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    search_summary_df = pd.DataFrame(search_records)
    search_summary_df["best_params"] = search_summary_df["best_params"].map(str)
    search_summary_df.to_csv(METRIC_DIR / "hyperparameter_search_summary.csv", index=False, encoding="utf-8-sig")

    all_importance_df = pd.concat(importance_records, ignore_index=True)
    all_importance_df.to_csv(IMPORTANCE_DIR / "all_models_feature_importance.csv", index=False, encoding="utf-8-sig")

    overall_importance_df = (
        all_importance_df.groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"), max_importance=("importance", "max"))
        .sort_values("mean_importance", ascending=False)
    )
    overall_importance_df["rank"] = np.arange(1, len(overall_importance_df) + 1)
    overall_importance_df.to_csv(IMPORTANCE_DIR / "overall_mean_feature_importance.csv", index=False, encoding="utf-8-sig")

    plot_metrics(metrics_df)
    best_row = metrics_df[metrics_df["split"] == "test"].sort_values("R2", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    plot_best_model_outputs(best_model_name, y_test.values, test_predictions[best_model_name])

    dataset_info = {
        "source_dataset": str(DATASET_CSV),
        "total_samples_after_dropna": len(train_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "split_mode": f"fixed {EXCEL_BATCH_COLUMN} in {TEST_BATCHES}",
        "input_features": len(feature_names),
        "best_model_by_test_R2": best_model_name,
        "best_model_test_R2": f"{best_row['R2']:.4f}",
        "best_model_test_MAE": f"{best_row['MAE']:.4f}",
        "best_model_test_RMSE": f"{best_row['RMSE']:.4f}",
    }
    write_markdown_report(metrics_df, best_model_name, dataset_info, model_times, search_summary_df)

    print("\n" + "=" * 70)
    print("[Complete] All beauty score regression models trained and explained")
    print(f"[Output] {OUTPUT_DIR}")
    print("\n[Test metrics]")
    print(metrics_df[metrics_df["split"] == "test"].sort_values("R2", ascending=False)[[
        "model", "MAE", "RMSE", "R2", "Acc@0.5score", "Acc@1.0score", "beautiful_accuracy", "beautiful_f1"
    ]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        regenerate_report_from_outputs()
    else:
        main()
