"""
Female/male beauty-score prediction with random split and paper-friendly visualizations.

Data policy:
  - Run female.csv and male.csv sequentially in one invocation.
  - Do not use supplementary data.
  - Randomly split with test_size=0.2 and random_state=42.

Outputs are written beneath the local outputs directory, with one report.md
at each feature-group output root. All figures have PNG and PDF versions except the numerous
SHAP interaction plots, which are PNG-only.
"""

from pathlib import Path
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import train_beauty_score_prediction_models as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_PRED_DIR = Path(__file__).resolve().parent

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SENSITIVITY_ROOT = OUTPUT_ROOT / "sensitivity_analysis_5_metrics"
DATASET_RUNS = [
    ("female", PROJECT_ROOT / "derived_data" / "female_50_features.csv"),
    ("male", PROJECT_ROOT / "derived_data" / "male_50_features.csv"),
]

# These globals are reassigned by configure_run_paths() before each sex-specific run.
DATASET_CSV = DATASET_RUNS[0][1]
OUTPUT_DIR = OUTPUT_ROOT / DATASET_RUNS[0][0]
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
MODEL_COMPARISON_DIR = ANALYSIS_DIR / "01_model_comparison"
BEST_MODEL_DIR = ANALYSIS_DIR / "02_best_model"
BEST_MODEL_PERFORMANCE_DIR = BEST_MODEL_DIR / "01_performance"
BEST_MODEL_VIS_DIR = BEST_MODEL_DIR / "02_shap"
TOP30_OVERALL_INTERRELATION_DIR = BEST_MODEL_DIR / "03_top30_correlation"
VIS_DIR = MODEL_COMPARISON_DIR
FEATURE_INTERRELATION_DIR = TOP30_OVERALL_INTERRELATION_DIR

TRAIN_DATASET_FILENAME = "random_split_train_dataset.csv"
TEST_DATASET_FILENAME = "random_split_test_dataset.csv"

MAX_SHAP_HEATMAP_FEATURES = 20
MAX_SHAP_INTERACTION_FEATURES = 10
MAX_WATERFALL_SAMPLES = 6
MAX_MODEL_AGNOSTIC_SHAP_BACKGROUND = 50
MODEL_AGNOSTIC_SHAP_NSAMPLES = 1000
MODEL_AGNOSTIC_SHAP_L1_REG = 0.0
TOP_OVERALL_CORRELATION_FEATURES = 30

MODEL_SELECTION_METRICS = [
    ("R2", False, 1.0),
    ("MAE", True, 1.0),
    ("RMSE", True, 1.0),
]
MODEL_SELECTION_DESCRIPTION = "综合排名：R2↑、MAE↓、RMSE↓ 等权平均名次"
SENSITIVITY_SELECTION_METRICS = [
    *MODEL_SELECTION_METRICS,
    ("Acc@0.5score", False, 1.0),
    ("Acc@1.0score", False, 1.0),
]
SENSITIVITY_SELECTION_DESCRIPTION = (
    "敏感性排名：R2↑、MAE↓、RMSE↓、Acc@0.5score↑、Acc@1.0score↑ 等权平均名次"
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ORIGINAL_PLT_SAVEFIG = plt.savefig


def savefig_with_pdf(filename, *args, **kwargs):
    """Save every pyplot PNG as PDF too, except bulk SHAP interaction plots."""
    ORIGINAL_PLT_SAVEFIG(filename, *args, **kwargs)
    path = Path(filename)
    is_bulk_interaction = "interactions" in {part.lower() for part in path.parts}
    if path.suffix.lower() == ".png" and not is_bulk_interaction:
        pdf_kwargs = dict(kwargs)
        pdf_kwargs.pop("format", None)
        ORIGINAL_PLT_SAVEFIG(path.with_suffix(".pdf"), *args, **pdf_kwargs)


plt.savefig = savefig_with_pdf


def configure_run_paths(dataset_csv: Path, output_dir: Path):
    global DATASET_CSV, OUTPUT_DIR, ANALYSIS_DIR, MODEL_COMPARISON_DIR
    global BEST_MODEL_DIR, BEST_MODEL_PERFORMANCE_DIR, VIS_DIR, BEST_MODEL_VIS_DIR
    global FEATURE_INTERRELATION_DIR, TOP30_OVERALL_INTERRELATION_DIR
    DATASET_CSV = Path(dataset_csv)
    OUTPUT_DIR = Path(output_dir)
    ANALYSIS_DIR = OUTPUT_DIR / "analysis"
    MODEL_COMPARISON_DIR = ANALYSIS_DIR / "01_model_comparison"
    BEST_MODEL_DIR = ANALYSIS_DIR / "02_best_model"
    BEST_MODEL_PERFORMANCE_DIR = BEST_MODEL_DIR / "01_performance"
    BEST_MODEL_VIS_DIR = BEST_MODEL_DIR / "02_shap"
    TOP30_OVERALL_INTERRELATION_DIR = BEST_MODEL_DIR / "03_top30_correlation"
    VIS_DIR = MODEL_COMPARISON_DIR
    FEATURE_INTERRELATION_DIR = TOP30_OVERALL_INTERRELATION_DIR
    configure_base_paths()


def configure_base_paths():
    base.DATASET_CSV = DATASET_CSV
    base.OUTPUT_DIR = OUTPUT_DIR
    base.MODEL_DIR = OUTPUT_DIR / "models"
    base.PRED_DIR = OUTPUT_DIR / "predictions"
    base.METRIC_DIR = OUTPUT_DIR / "metrics"
    base.IMPORTANCE_DIR = OUTPUT_DIR / "_unused_feature_importance"
    base.PLOT_DIR = MODEL_COMPARISON_DIR
    base.TRAIN_DATASET_FILENAME = TRAIN_DATASET_FILENAME
    base.TEST_DATASET_FILENAME = TEST_DATASET_FILENAME
    base.TOP_N_FEATURES = 10_000


def ensure_extra_dirs():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_VIS_DIR.mkdir(parents=True, exist_ok=True)
    TOP30_OVERALL_INTERRELATION_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in [
        VIS_DIR / "*.png",
        VIS_DIR / "*.pdf",
        VIS_DIR / "*.csv",
        BEST_MODEL_PERFORMANCE_DIR / "*.png",
        BEST_MODEL_PERFORMANCE_DIR / "*.pdf",
        BEST_MODEL_PERFORMANCE_DIR / "*.csv",
        BEST_MODEL_VIS_DIR / "*.png",
        BEST_MODEL_VIS_DIR / "*.pdf",
        BEST_MODEL_VIS_DIR / "*.csv",
        BEST_MODEL_VIS_DIR / "interactions" / "*.png",
        BEST_MODEL_VIS_DIR / "interactions" / "*.pdf",
        BEST_MODEL_VIS_DIR / "waterfall" / "*.png",
        BEST_MODEL_VIS_DIR / "waterfall" / "*.pdf",
        TOP30_OVERALL_INTERRELATION_DIR / "*.png",
        TOP30_OVERALL_INTERRELATION_DIR / "*.pdf",
        TOP30_OVERALL_INTERRELATION_DIR / "*.csv",
    ]:
        for path in pattern.parent.glob(pattern.name):
            path.unlink()


def load_dataset_random_only():
    df = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    feature_columns = [col for col in df.columns if col.startswith(base.FEATURE_PREFIX)]
    if not feature_columns:
        raise ValueError(f"{DATASET_CSV} 中没有找到 {base.FEATURE_PREFIX} 开头的 measurement 特征列")
    if base.TARGET_COLUMN not in df.columns:
        raise ValueError(f"{DATASET_CSV} 缺少回归目标列: {base.TARGET_COLUMN}")

    df = df.dropna(subset=feature_columns + [base.TARGET_COLUMN]).reset_index(drop=True)
    if base.AUX_CLASS_COLUMN not in df.columns:
        df[base.AUX_CLASS_COLUMN] = (df[base.TARGET_COLUMN] >= base.BEAUTIFUL_THRESHOLD).astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=base.TEST_SIZE,
        random_state=base.RANDOM_STATE,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True).copy()
    test_df = test_df.reset_index(drop=True).copy()
    train_df["split_policy_20260803"] = "random_train_original_only"
    test_df["split_policy_20260803"] = "random_test_original_only"

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    y_train = train_df[base.TARGET_COLUMN].astype(float).copy()
    y_test = test_df[base.TARGET_COLUMN].astype(float).copy()

    id_columns = [col for col in base.ID_COLUMNS + ["split_policy_20260803"] if col in train_df.columns or col in test_df.columns]
    train_ids = train_df[[col for col in id_columns if col in train_df.columns]].copy()
    test_ids = test_df[[col for col in id_columns if col in test_df.columns]].copy()
    return train_df, test_df, train_ids, test_ids, X_train, X_test, y_train, y_test, feature_columns


def model_selection_table(metrics_df, selection_metrics=None):
    selection_metrics = selection_metrics or MODEL_SELECTION_METRICS
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    rank_columns = []
    total_weight = 0.0
    weighted_rank_sum = pd.Series(0.0, index=test_df.index)
    for metric, ascending, weight in selection_metrics:
        rank_col = f"selection_rank__{metric}"
        test_df[rank_col] = test_df[metric].rank(ascending=ascending, method="average")
        weighted_rank_sum = weighted_rank_sum + test_df[rank_col] * weight
        total_weight += weight
        rank_columns.append(rank_col)

    test_df["model_selection_score"] = weighted_rank_sum / total_weight
    test_df = test_df.sort_values(
        ["model_selection_score", "R2", "MAE", "RMSE"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    test_df["model_selection_rank"] = np.arange(1, len(test_df) + 1)
    return test_df[["model", "model_selection_rank", "model_selection_score", *rank_columns]]


def metric_sort_order(metrics_df, selection_metrics=None):
    return model_selection_table(metrics_df, selection_metrics)["model"].tolist()


def plot_regression_metric_bars(metrics_df):
    """Plot regression metrics only; auxiliary threshold classification is CSV-only."""
    test_df = metrics_df[metrics_df["split"] == "test"].copy()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=test_df.sort_values("R2", ascending=False), x="R2", y="model")
    plt.title("Beauty Score Regression Test R2 by Model")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_r2_by_model.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=test_df.sort_values("MAE"), x="MAE", y="model")
    plt.title("Beauty Score Regression Test MAE by Model")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_mae_by_model.png", dpi=300)
    plt.close()


def plot_model_metric_visualizations(metrics_df, selection_metrics=None):
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    model_order = metric_sort_order(metrics_df, selection_metrics)
    test_df["model"] = pd.Categorical(test_df["model"], categories=model_order, ordered=True)
    test_df = test_df.sort_values("model")

    plt.figure(figsize=(13, 6))
    sns.lineplot(data=test_df, x="model", y="MAE", marker="o", linewidth=2, label="MAE")
    sns.lineplot(data=test_df, x="model", y="RMSE", marker="o", linewidth=2, label="RMSE")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Error")
    plt.xlabel("Model")
    plt.title("Test Error Metrics by Model")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_mae_rmse_curves_by_model.png", dpi=300)
    plt.close()

    plt.figure(figsize=(13, 6))
    sns.lineplot(data=test_df, x="model", y="R2", marker="o", linewidth=2, label="R2")
    sns.lineplot(data=test_df, x="model", y="Acc@0.5score", marker="o", linewidth=2, label="Acc <= 0.5")
    sns.lineplot(data=test_df, x="model", y="Acc@1.0score", marker="o", linewidth=2, label="Acc <= 1.0")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Test R2 and Score-Accuracy Curves by Model")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_r2_accuracy_curves_by_model.png", dpi=300)
    plt.close()

    long_metrics = test_df.melt(
        id_vars="model",
        value_vars=["MAE", "RMSE", "R2", "Acc@0.5score", "Acc@1.0score"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(14, 6))
    sns.barplot(data=long_metrics, x="model", y="value", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Metric value")
    plt.title("Test Metric Comparison Across Models")
    plt.legend(loc="upper right", ncol=3)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_metrics_grouped_bar_by_model.png", dpi=300)
    plt.close()

    heatmap_df = test_df.set_index("model")[["MAE", "RMSE", "R2", "Acc@0.5score", "Acc@1.0score"]]
    plt.figure(figsize=(9, max(5, 0.38 * len(heatmap_df))))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5)
    plt.title("Test Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_metrics_heatmap_by_model.png", dpi=300)
    plt.close()

    train_r2 = metrics_df[metrics_df["split"] == "train"][["model", "R2"]].rename(columns={"R2": "train_R2"})
    gap_df = test_df[["model", "R2"]].rename(columns={"R2": "test_R2"}).merge(train_r2, on="model", how="left")
    gap_df["R2_gap"] = gap_df["train_R2"] - gap_df["test_R2"]
    gap_long = gap_df.melt(id_vars="model", value_vars=["train_R2", "test_R2", "R2_gap"], var_name="metric", value_name="value")
    plt.figure(figsize=(14, 6))
    sns.barplot(data=gap_long, x="model", y="value", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Model")
    plt.ylabel("R2")
    plt.title("Train/Test R2 and Generalization Gap")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "train_test_r2_gap_by_model.png", dpi=300)
    plt.close()


def plot_prediction_error_visualizations(metrics_df, y_true, predictions_by_model, selection_metrics=None):
    model_order = metric_sort_order(metrics_df, selection_metrics)
    frames = []
    for model_name in model_order:
        if model_name not in predictions_by_model:
            continue
        prediction = np.asarray(predictions_by_model[model_name], dtype=float)
        pred_df = pd.DataFrame({
            "true_beauty_score": np.asarray(y_true, dtype=float),
            "pred_beauty_score": prediction,
        })
        pred_df["error_beauty_score"] = pred_df["pred_beauty_score"] - pred_df["true_beauty_score"]
        pred_df["model"] = model_name
        pred_df["abs_error_beauty_score"] = pred_df["error_beauty_score"].abs()
        frames.append(pred_df)
    if not frames:
        return

    pred_all = pd.concat(frames, ignore_index=True)
    pred_all["model"] = pd.Categorical(pred_all["model"], categories=model_order, ordered=True)
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=pred_all, x="model", y="abs_error_beauty_score", showfliers=False)
    sns.stripplot(data=pred_all, x="model", y="abs_error_beauty_score", color="black", alpha=0.18, size=2)
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Absolute error")
    plt.title("Test Absolute Error Distribution by Model")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_absolute_error_distribution_by_model.png", dpi=300)
    plt.close()

    plt.figure(figsize=(14, 6))
    sns.violinplot(data=pred_all, x="model", y="error_beauty_score", inner="quartile", cut=0)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Prediction error")
    plt.title("Test Signed Error Distribution by Model")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "test_signed_error_violin_by_model.png", dpi=300)
    plt.close()


def clean_feature_name(name: str) -> str:
    return str(name).replace(base.FEATURE_PREFIX, "")


def clean_feature_names(feature_names):
    return [clean_feature_name(name) for name in feature_names]


def _save_correlation_figure(out_path: Path):
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _complete_correlation_submatrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable features until scipy receives a finite square matrix."""
    valid = corr.replace([np.inf, -np.inf], np.nan).copy()
    all_nan = valid.index[valid.isna().all(axis=1)]
    valid = valid.drop(index=all_nan, columns=all_nan, errors="ignore")
    while valid.isna().to_numpy().any() and len(valid) >= 2:
        missing = valid.isna().sum(axis=0) + valid.isna().sum(axis=1)
        worst_feature = missing.idxmax()
        valid = valid.drop(index=worst_feature, columns=worst_feature)
    return valid


def plot_top30_overall_correlation_clustermap(best_model_name, X_all, feature_names, feature_rank):
    """Select Top30 from the one best-model SHAP rank, then cluster Pearson profiles."""
    raw_feature_lookup = {}
    for raw_feature in feature_names:
        raw_feature_lookup[str(raw_feature)] = raw_feature
        raw_feature_lookup[clean_feature_name(raw_feature)] = raw_feature

    selected_rows = []
    selected_raw_features = []
    seen = set()
    for _, row in feature_rank.iterrows():
        importance_feature = str(row["feature"])
        raw_feature = raw_feature_lookup.get(importance_feature)
        if raw_feature is None or raw_feature in seen:
            continue
        seen.add(raw_feature)
        selected_raw_features.append(raw_feature)
        selected_row = row.to_dict()
        selected_row["raw_feature"] = raw_feature
        selected_row["display_feature"] = clean_feature_name(raw_feature)
        selected_rows.append(selected_row)
        if len(selected_raw_features) >= TOP_OVERALL_CORRELATION_FEATURES:
            break

    if len(selected_raw_features) < 2:
        print("[Warning] 可用于 Top30 相关聚类的最优模型重要特征不足 2 个，已跳过。")
        return

    top_features_df = pd.DataFrame(selected_rows)
    top_features_df.insert(0, "top30_selection_order", np.arange(1, len(top_features_df) + 1))
    correlation_data = X_all[selected_raw_features].copy()
    usable_raw_features = [
        feature for feature in selected_raw_features
        if correlation_data[feature].notna().sum() >= 2
        and correlation_data[feature].nunique(dropna=True) >= 2
    ]
    top_features_df["used_in_correlation"] = top_features_df["raw_feature"].isin(usable_raw_features)
    top_features_df.to_csv(
        TOP30_OVERALL_INTERRELATION_DIR / "top30_overall_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if len(usable_raw_features) < 2:
        print("[Warning] Top30 中非恒定、有效的输入特征不足 2 个，已跳过相关聚类图。")
        return

    display_map = {feature: clean_feature_name(feature) for feature in usable_raw_features}
    corr = (
        correlation_data[usable_raw_features]
        .rename(columns=display_map)
        .corr(method="pearson")
    )
    corr = _complete_correlation_submatrix(corr)
    corr.to_csv(
        TOP30_OVERALL_INTERRELATION_DIR / "top30_overall_correlation_matrix.csv",
        encoding="utf-8-sig",
    )
    if len(corr) < 2:
        print("[Warning] Top30 Pearson 相关矩阵没有足够的有限值用于聚类，已跳过绘图。")
        return

    size = max(10, 0.46 * len(corr))
    plt.figure(figsize=(size, size * 0.92))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=False,
        cbar_kws={"label": "Pearson r"},
        annot_kws={"size": 7},
    )
    plt.title(f"{best_model_name}: Top{len(corr)} Important Input Feature Correlations")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    _save_correlation_figure(
        TOP30_OVERALL_INTERRELATION_DIR / "top30_overall_correlation_heatmap.png"
    )

    grid = sns.clustermap(
        corr,
        method="average",
        metric="euclidean",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(size, size),
        cbar_kws={"label": "Pearson r"},
    )
    grid.fig.suptitle(
        f"{best_model_name}: Top{len(corr)} Correlation-Profile Clusters",
        y=1.02,
        fontsize=13,
    )
    out_path = TOP30_OVERALL_INTERRELATION_DIR / "top30_overall_correlation_clustermap.png"
    grid.savefig(out_path, dpi=300, bbox_inches="tight")
    grid.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(grid.fig)
    print(f"[Interrelation] 已生成最优模型 Top{len(corr)} Pearson 相关热力图与层次聚类图: {out_path}")


def best_model_file_prefix(best_model_name: str) -> str:
    safe_name = base.sanitize_filename(str(best_model_name))
    return safe_name if safe_name else "best_model"


def fix_unicode_minus(fig=None):
    fig = fig or plt.gcf()
    fig.canvas.draw()
    minus_chars = ["−", "–", "—", "﹣", "－"]
    for text_obj in fig.findobj(match=Text):
        text = text_obj.get_text()
        fixed = text
        for minus_char in minus_chars:
            fixed = fixed.replace(minus_char, "-")
        if fixed != text:
            text_obj.set_text(fixed)
    for ax in fig.axes:
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            text = label.get_text()
            fixed = text
            for minus_char in minus_chars:
                fixed = fixed.replace(minus_char, "-")
            if fixed != text:
                label.set_text(fixed)
    fig.canvas.draw()


def plot_best_model_regression_performance(best_model_name, y_test, y_pred):
    """Best-model regression scatter only; no auxiliary confusion matrix."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.75, edgecolor="k", linewidth=0.3)
    min_v = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel("True beauty score")
    plt.ylabel("Predicted beauty score")
    plt.title(f"Best Model Test Scatter: {best_model_name}")
    plt.tight_layout()
    plt.savefig(BEST_MODEL_PERFORMANCE_DIR / f"{best_model_name}_test_scatter.png", dpi=300)
    plt.close()


def get_sample_labels(test_ids: pd.DataFrame) -> list[str]:
    if "样本名" in test_ids.columns:
        values = test_ids["样本名"]
    elif "姓名" in test_ids.columns:
        values = test_ids["姓名"]
    else:
        values = pd.Series(np.arange(len(test_ids)), index=test_ids.index).astype(str)
    labels = []
    for idx, value in enumerate(values):
        text = str(value)
        labels.append(text if text and text != "nan" else f"sample_{idx}")
    return labels


def compute_shap_values(best_model, best_model_name, X_train_scaled, X_test_scaled):
    if base.shap is None:
        raise RuntimeError("shap is not installed")
    background = X_train_scaled.sample(min(base.SHAP_BACKGROUND_SIZE, len(X_train_scaled)), random_state=base.RANDOM_STATE)
    eval_df = X_test_scaled.copy()

    try:
        if hasattr(best_model, "coef_"):
            explainer = base.shap.LinearExplainer(best_model, background)
            shap_values = explainer.shap_values(eval_df)
            expected_value = explainer.expected_value
            method = "LinearExplainer"
        else:
            explainer = base.shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(eval_df)
            expected_value = explainer.expected_value
            method = "TreeExplainer"
    except Exception as error:
        background = X_train_scaled.sample(
            min(MAX_MODEL_AGNOSTIC_SHAP_BACKGROUND, len(X_train_scaled)),
            random_state=base.RANDOM_STATE,
        )
        eval_df = X_test_scaled.copy()

        def predict_fn(values):
            values_df = pd.DataFrame(values, columns=X_train_scaled.columns)
            return np.asarray(best_model.predict(values_df)).reshape(-1)

        print(
            f"[SHAP] {best_model_name}: Tree/Linear explainer unsupported ({error}); "
            f"fallback to KernelExplainer with {len(background)} background and {len(eval_df)} eval samples"
        )
        explainer = base.shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(
            eval_df,
            nsamples=MODEL_AGNOSTIC_SHAP_NSAMPLES,
            l1_reg=MODEL_AGNOSTIC_SHAP_L1_REG,
        )
        expected_value = explainer.expected_value
        method = "KernelExplainer"

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    expected_value = np.asarray(expected_value).reshape(-1)[0]
    print(f"[SHAP] {best_model_name}: {method}, shap_values shape={shap_values.shape}")
    return shap_values, float(expected_value), eval_df, method


def save_shap_tables(shap_values, X_test_scaled, sample_labels, feature_names, best_model_name, method):
    model_prefix = best_model_file_prefix(best_model_name)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, "sample", sample_labels)
    shap_df.to_csv(BEST_MODEL_VIS_DIR / f"{model_prefix}_test_shap_values_all_features.csv", index=False, encoding="utf-8-sig")

    feature_rank = pd.DataFrame({
        "feature": feature_names,
        "original_index": np.arange(len(feature_names)),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    feature_rank["rank"] = np.arange(1, len(feature_rank) + 1)
    feature_rank["explain_method"] = method
    feature_rank["explained_test_samples"] = len(shap_values)
    feature_rank.to_csv(BEST_MODEL_VIS_DIR / f"{model_prefix}_shap_feature_rank.csv", index=False, encoding="utf-8-sig")

    X_export = X_test_scaled.copy()
    X_export.insert(0, "sample", sample_labels)
    X_export.to_csv(BEST_MODEL_VIS_DIR / f"{model_prefix}_test_scaled_features.csv", index=False, encoding="utf-8-sig")
    return feature_rank


def plot_best_model_shap_heatmap(shap_values, sample_labels, feature_rank, best_model_name):
    model_prefix = best_model_file_prefix(best_model_name)
    top_features = feature_rank.head(MAX_SHAP_HEATMAP_FEATURES).copy()
    top_idx = top_features["original_index"].astype(int).tolist()
    heat_df = pd.DataFrame(
        shap_values[:, top_idx],
        columns=top_features["feature"].tolist(),
    )

    plt.figure(figsize=(15, max(8, 0.055 * len(heat_df))))
    sns.heatmap(heat_df, cmap="coolwarm", center=0, yticklabels=False, cbar_kws={"label": "SHAP value"})
    plt.xlabel("Top SHAP features")
    plt.ylabel("Test sample index")
    plt.title(f"{best_model_name} SHAP Heatmap: Test Samples x Important Features")
    plt.tight_layout()
    plt.savefig(BEST_MODEL_VIS_DIR / f"{model_prefix}_test_sample_feature_shap_heatmap.png", dpi=300)
    plt.close()


def plot_best_model_shap_summary(shap_values, X_test_scaled, feature_names, best_model_name):
    model_prefix = best_model_file_prefix(best_model_name)
    if base.shap is None:
        return
    plt.figure()
    base.shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, max_display=len(feature_names), show=False)
    plt.gcf().suptitle(f"{best_model_name} SHAP Summary for All Input Features", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(BEST_MODEL_VIS_DIR / f"{model_prefix}_shap_summary_all_features.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    base.shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, plot_type="bar", max_display=len(feature_names), show=False)
    plt.gcf().suptitle(f"{best_model_name} Mean |SHAP| Importance for All Input Features", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(BEST_MODEL_VIS_DIR / f"{model_prefix}_shap_bar_all_features.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_best_model_interactions(shap_values, X_test_scaled, feature_rank, best_model_name):
    model_prefix = best_model_file_prefix(best_model_name)
    if base.shap is None:
        return
    top_features = feature_rank.head(MAX_SHAP_INTERACTION_FEATURES).copy()
    interaction_dir = BEST_MODEL_VIS_DIR / "interactions"
    interaction_dir.mkdir(parents=True, exist_ok=True)

    for _, row_1 in top_features.iterrows():
        for _, row_2 in top_features.iterrows():
            if int(row_1["original_index"]) >= int(row_2["original_index"]):
                continue
            feature_1 = row_1["feature"]
            feature_2 = row_2["feature"]
            index_1 = int(row_1["original_index"])
            index_2 = int(row_2["original_index"])
            safe_name = base.sanitize_filename(f"{feature_1}_vs_{feature_2}")
            try:
                plt.figure(figsize=(6.5, 5.5))
                base.shap.dependence_plot(
                    ind=index_1,
                    shap_values=shap_values,
                    features=X_test_scaled.values,
                    feature_names=list(X_test_scaled.columns),
                    interaction_index=index_2,
                    show=False,
                )
                plt.title(f"{best_model_name} SHAP Interaction: {feature_1} vs {feature_2}")
                plt.tight_layout()
                plt.savefig(interaction_dir / f"{model_prefix}_shap_interaction_{safe_name}.png", dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as error:
                plt.close()
                print(f"[Warning] interaction plot failed for {feature_1} vs {feature_2}: {error}")


def plot_best_model_waterfalls(shap_values, expected_value, X_test_scaled, sample_labels, feature_names, y_true, y_pred, best_model_name):
    model_prefix = best_model_file_prefix(best_model_name)
    if base.shap is None:
        return
    errors = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    candidate_indices = list(np.argsort(errors)[-3:][::-1]) + list(np.argsort(errors)[:3])
    candidate_indices = list(dict.fromkeys(candidate_indices))[:MAX_WATERFALL_SAMPLES]
    waterfall_dir = BEST_MODEL_VIS_DIR / "waterfall"
    waterfall_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in candidate_indices:
        label = base.sanitize_filename(sample_labels[sample_idx]) or f"sample_{sample_idx}"
        explanation = base.shap.Explanation(
            values=shap_values[sample_idx],
            base_values=expected_value,
            feature_names=feature_names,
        )
        plt.figure(figsize=(9, 7))
        base.shap.waterfall_plot(explanation, max_display=18, show=False)
        plt.title(
            f"{best_model_name} SHAP Waterfall: {sample_labels[sample_idx]} | "
            f"true={float(y_true[sample_idx]):.2f}, pred={float(y_pred[sample_idx]):.2f}"
        )
        fix_unicode_minus(plt.gcf())
        plt.tight_layout()
        plt.savefig(waterfall_dir / f"{model_prefix}_waterfall_{sample_idx:03d}_{label}.png", dpi=300, bbox_inches="tight")
        plt.close()


def run_best_model_shap_visualizations(best_model_name, best_model, X_train_scaled, X_test_scaled, test_ids, feature_names, y_true, y_pred):
    try:
        shap_values, expected_value, eval_df, method = compute_shap_values(best_model, best_model_name, X_train_scaled, X_test_scaled)
    except Exception as error:
        print(f"[Warning] 最优模型 SHAP 可视化失败: {error}")
        raise

    eval_display_df = eval_df.copy()
    eval_display_df.columns = feature_names
    label_series = pd.Series(get_sample_labels(test_ids), index=X_test_scaled.index)
    sample_labels = label_series.loc[eval_df.index].tolist()
    y_true_eval = pd.Series(np.asarray(y_true), index=X_test_scaled.index).loc[eval_df.index].to_numpy()
    y_pred_eval = pd.Series(np.asarray(y_pred), index=X_test_scaled.index).loc[eval_df.index].to_numpy()
    feature_rank = save_shap_tables(shap_values, eval_display_df, sample_labels, feature_names, best_model_name, method)
    plot_best_model_shap_heatmap(shap_values, sample_labels, feature_rank, best_model_name)
    plot_best_model_shap_summary(shap_values, eval_display_df, feature_names, best_model_name)
    plot_best_model_interactions(shap_values, eval_display_df, feature_rank, best_model_name)
    plot_best_model_waterfalls(shap_values, expected_value, eval_display_df, sample_labels, feature_names, y_true_eval, y_pred_eval, best_model_name)
    return feature_rank


def run_single_dataset(sex_label: str, dataset_csv: Path):
    configure_run_paths(dataset_csv, OUTPUT_ROOT / sex_label)
    base.ensure_dirs()
    ensure_extra_dirs()
    inner_report = OUTPUT_DIR / "report.md"
    if inner_report.exists():
        inner_report.unlink()

    train_df, test_df, train_ids, test_ids, X_train, X_test, y_train, y_test, feature_names = load_dataset_random_only()
    display_feature_names = clean_feature_names(feature_names)
    base.TOP_N_FEATURES = len(feature_names)
    X_train_scaled, X_test_scaled, y_train_scaled, x_scaler, y_scaler = base.scale_datasets(X_train, X_test, y_train)
    print("=" * 70)
    print(f"Beauty score regression: {sex_label} random split + rich visualizations")
    print(f"Dataset: {DATASET_CSV}")
    print(f"Random split: train {len(X_train)}, test {len(X_test)}, test_size={base.TEST_SIZE}, random_state={base.RANDOM_STATE}")
    print("Supplementary data: not used")
    print(f"Input measurement features: {len(feature_names)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    train_df.to_csv(OUTPUT_DIR / TRAIN_DATASET_FILENAME, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / TEST_DATASET_FILENAME, index=False, encoding="utf-8-sig")
    joblib.dump(x_scaler, base.MODEL_DIR / "input_feature_standard_scaler.joblib")
    joblib.dump(y_scaler, base.MODEL_DIR / "target_score_standard_scaler.joblib")

    model_specs = base.build_models()
    metric_records = []
    model_times = {}
    test_predictions = {}
    trained_models = {}
    search_records = []

    for model_name, spec in model_specs.items():
        print(f"\n[Optimize/Train] {model_name}")
        start_time = base.time.time()
        model, search_record = base.tune_or_fit_model(model_name, spec, X_train_scaled, y_train_scaled)
        elapsed = base.time.time() - start_time
        model_times[model_name] = elapsed
        search_records.append(search_record)
        trained_models[model_name] = model
        if pd.isna(search_record["best_cv_rmse_scaled"]):
            print(f"[Done] {model_name} fitted in {elapsed:.2f}s ({search_record['search_type']})")
        else:
            print(f"[Done] {model_name} search+fit time: {elapsed:.2f}s, best CV RMSE(scaled): {search_record['best_cv_rmse_scaled']:.4f}")
            print(f"[Best Params] {search_record['best_params']}")

        train_pred = base.inverse_transform_target(y_scaler, model.predict(X_train_scaled))
        test_pred = base.inverse_transform_target(y_scaler, model.predict(X_test_scaled))
        test_predictions[model_name] = test_pred

        metric_records.append(base.evaluate_predictions(model_name, y_train.values, train_pred, "train"))
        metric_records.append(base.evaluate_predictions(model_name, y_test.values, test_pred, "test"))

        joblib.dump(model, base.MODEL_DIR / f"{model_name}.joblib")

    metrics_df = pd.DataFrame(metric_records)
    selection_df = model_selection_table(metrics_df)
    metrics_df = metrics_df.merge(
        selection_df[["model", "model_selection_rank", "model_selection_score"]],
        on="model",
        how="left",
    )
    metrics_df.to_csv(base.METRIC_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(base.METRIC_DIR / "model_selection_ranking.csv", index=False, encoding="utf-8-sig")

    search_summary_df = pd.DataFrame(search_records)
    search_summary_df["best_params"] = search_summary_df["best_params"].map(str)
    search_summary_df.to_csv(base.METRIC_DIR / "hyperparameter_search_summary.csv", index=False, encoding="utf-8-sig")

    plot_regression_metric_bars(metrics_df)
    plot_model_metric_visualizations(metrics_df)
    plot_prediction_error_visualizations(metrics_df, y_test.values, test_predictions)

    best_model_name = selection_df.iloc[0]["model"]
    best_row = metrics_df[(metrics_df["split"] == "test") & (metrics_df["model"] == best_model_name)].iloc[0]
    best_model_name = best_row["model"]
    plot_best_model_regression_performance(best_model_name, y_test.values, test_predictions[best_model_name])
    feature_rank = run_best_model_shap_visualizations(
        best_model_name,
        trained_models[best_model_name],
        X_train_scaled,
        X_test_scaled,
        test_ids,
        display_feature_names,
        y_test.values,
        test_predictions[best_model_name],
    )
    X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    plot_top30_overall_correlation_clustermap(best_model_name, X_all, feature_names, feature_rank)
    if base.IMPORTANCE_DIR.exists() and not any(base.IMPORTANCE_DIR.iterdir()):
        base.IMPORTANCE_DIR.rmdir()
    if base.PRED_DIR.exists() and not any(base.PRED_DIR.iterdir()):
        base.PRED_DIR.rmdir()

    dataset_info = {
        "sex": sex_label,
        "source_dataset": str(DATASET_CSV),
        "total_samples_after_dropna": len(train_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "split_mode": f"random split only, test_size={base.TEST_SIZE}, random_state={base.RANDOM_STATE}; no supplementary data",
        "input_features": len(feature_names),
        "best_model_by_test_R2": best_model_name,
        "best_model_selection_policy": MODEL_SELECTION_DESCRIPTION,
        "best_model_selection_score": f"{best_row['model_selection_score']:.4f}",
        "best_model_test_R2": f"{best_row['R2']:.4f}",
        "best_model_test_MAE": f"{best_row['MAE']:.4f}",
        "best_model_test_RMSE": f"{best_row['RMSE']:.4f}",
    }

    print("\n" + "=" * 70)
    print(f"[Complete] {sex_label} random-only beauty score models trained with rich visualizations")
    print(f"[Output] {OUTPUT_DIR}")
    print("\n[Test metrics]")
    print(metrics_df[metrics_df["split"] == "test"].sort_values("model_selection_rank")[[
        "model", "model_selection_rank", "model_selection_score", "MAE", "RMSE", "R2", "Acc@0.5score", "Acc@1.0score", "beautiful_accuracy", "beautiful_f1"
    ]].to_string(index=False))
    print("=" * 70)

    return {
        "sex": sex_label,
        "output_dir": OUTPUT_DIR,
        "dataset_info": dataset_info,
        "metrics_df": metrics_df,
        "selection_df": selection_df,
        "search_summary_df": search_summary_df,
        "model_times": model_times,
        "best_model_name": best_model_name,
        "feature_rank": feature_rank,
    }


def verify_pdf_coverage(output_dir: Path):
    missing = []
    for png_path in output_dir.rglob("*.png"):
        if "interactions" in {part.lower() for part in png_path.parts}:
            continue
        if not png_path.with_suffix(".pdf").exists():
            missing.append(png_path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise RuntimeError(f"以下非 interaction 图片缺少 PDF（共 {len(missing)} 个）:\n{preview}")


def report_float(value, digits=4):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def build_sensitivity_analysis_for_dataset(sex_label: str, dataset_csv: Path):
    """Build a self-contained five-metric sensitivity folder from saved primary outputs."""
    primary_output_dir = OUTPUT_ROOT / sex_label
    sensitivity_output_dir = SENSITIVITY_ROOT / sex_label
    metrics_path = primary_output_dir / "metrics" / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少正式分析结果，无法生成敏感性分析: {metrics_path}")

    sensitivity_output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(primary_output_dir, sensitivity_output_dir, dirs_exist_ok=True)
    configure_run_paths(dataset_csv, sensitivity_output_dir)
    base.ensure_dirs()
    ensure_extra_dirs()

    metrics_df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    official_selection_df = pd.read_csv(
        primary_output_dir / "metrics" / "model_selection_ranking.csv",
        encoding="utf-8-sig",
    )
    raw_metrics_df = metrics_df.drop(
        columns=["model_selection_rank", "model_selection_score"],
        errors="ignore",
    )
    sensitivity_selection_df = model_selection_table(raw_metrics_df, SENSITIVITY_SELECTION_METRICS)
    sensitivity_metrics_df = raw_metrics_df.merge(
        sensitivity_selection_df[["model", "model_selection_rank", "model_selection_score"]],
        on="model",
        how="left",
    )
    sensitivity_metrics_df.to_csv(
        base.METRIC_DIR / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_selection_df.to_csv(
        base.METRIC_DIR / "model_selection_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    official_selection_df.to_csv(
        base.METRIC_DIR / "model_selection_ranking_3_metrics_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ranking_comparison_df = official_selection_df[[
        "model", "model_selection_rank", "model_selection_score"
    ]].rename(columns={
        "model_selection_rank": "official_3_metrics_rank",
        "model_selection_score": "official_3_metrics_score",
    }).merge(
        sensitivity_selection_df[["model", "model_selection_rank", "model_selection_score"]].rename(columns={
            "model_selection_rank": "sensitivity_5_metrics_rank",
            "model_selection_score": "sensitivity_5_metrics_score",
        }),
        on="model",
        how="outer",
    ).sort_values("sensitivity_5_metrics_rank")
    ranking_comparison_df.to_csv(
        base.METRIC_DIR / "ranking_comparison_3_vs_5_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "analysis": "sensitivity_analysis_5_metrics",
        "selection_policy": SENSITIVITY_SELECTION_DESCRIPTION,
        "ranking_formula": "mean(rank_R2, rank_MAE, rank_RMSE, rank_Acc@0.5score, rank_Acc@1.0score)",
        "source_primary_output": str(primary_output_dir),
    }]).to_csv(base.METRIC_DIR / "sensitivity_analysis_policy.csv", index=False, encoding="utf-8-sig")

    plot_regression_metric_bars(sensitivity_metrics_df)
    plot_model_metric_visualizations(sensitivity_metrics_df, SENSITIVITY_SELECTION_METRICS)
    best_model_name = str(sensitivity_selection_df.iloc[0]["model"])
    train_df = pd.read_csv(sensitivity_output_dir / TRAIN_DATASET_FILENAME, encoding="utf-8-sig")
    test_df = pd.read_csv(sensitivity_output_dir / TEST_DATASET_FILENAME, encoding="utf-8-sig")
    feature_names = [column for column in train_df.columns if column.startswith(base.FEATURE_PREFIX)]
    display_feature_names = clean_feature_names(feature_names)
    X_train = train_df[feature_names].copy()
    X_test = test_df[feature_names].copy()
    y_test = test_df[base.TARGET_COLUMN].astype(float).copy()
    x_scaler = joblib.load(base.MODEL_DIR / "input_feature_standard_scaler.joblib")
    y_scaler = joblib.load(base.MODEL_DIR / "target_score_standard_scaler.joblib")
    X_train_scaled = pd.DataFrame(
        x_scaler.transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        x_scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )
    sensitivity_predictions = {}
    loaded_models = {}
    for model_name in sensitivity_selection_df["model"].astype(str):
        model = joblib.load(base.MODEL_DIR / f"{model_name}.joblib")
        loaded_models[model_name] = model
        sensitivity_predictions[model_name] = base.inverse_transform_target(
            y_scaler,
            model.predict(X_test_scaled),
        )
    plot_prediction_error_visualizations(
        sensitivity_metrics_df,
        y_test.to_numpy(),
        sensitivity_predictions,
        SENSITIVITY_SELECTION_METRICS,
    )
    best_model = loaded_models[best_model_name]
    y_pred = sensitivity_predictions[best_model_name]
    split_policy_columns = [column for column in test_df.columns if column.startswith("split_policy_")]
    id_columns = [
        column for column in [*base.ID_COLUMNS, *split_policy_columns]
        if column in test_df.columns
    ]
    test_ids = test_df[id_columns].copy()

    plot_best_model_regression_performance(best_model_name, y_test.to_numpy(), y_pred)
    feature_rank = run_best_model_shap_visualizations(
        best_model_name,
        best_model,
        X_train_scaled,
        X_test_scaled,
        test_ids,
        display_feature_names,
        y_test.to_numpy(),
        y_pred,
    )
    X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    plot_top30_overall_correlation_clustermap(best_model_name, X_all, feature_names, feature_rank)
    if base.IMPORTANCE_DIR.exists() and not any(base.IMPORTANCE_DIR.iterdir()):
        base.IMPORTANCE_DIR.rmdir()
    if base.PRED_DIR.exists() and not any(base.PRED_DIR.iterdir()):
        base.PRED_DIR.rmdir()
    verify_pdf_coverage(sensitivity_output_dir)

    official_best_model = str(official_selection_df.iloc[0]["model"])
    sensitivity_best_row = sensitivity_metrics_df[
        (sensitivity_metrics_df["split"] == "test")
        & (sensitivity_metrics_df["model"] == best_model_name)
    ].iloc[0]
    print(
        f"[Sensitivity] {sex_label}: official={official_best_model}, "
        f"five-metric={best_model_name}, output={sensitivity_output_dir}"
    )
    return {
        "sex": sex_label,
        "output_dir": sensitivity_output_dir,
        "official_best_model": official_best_model,
        "best_model_name": best_model_name,
        "metrics_df": sensitivity_metrics_df,
        "selection_df": sensitivity_selection_df,
        "ranking_comparison_df": ranking_comparison_df,
        "best_row": sensitivity_best_row,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "input_features": len(feature_names),
    }


def write_sensitivity_report(sensitivity_results):
    lines = []
    lines.append("# 五指标模型选择敏感性分析\n\n")
    lines.append("本目录是独立于正式三指标分析的敏感性输出。female 和 male 均复用正式分析已经训练完成的模型与相同随机划分；不重新训练，只改变模型选择排名。制图所需预测值从已保存模型即时重新计算，不保存逐样本预测文件。\n\n")
    lines.append("## 排名方法\n\n")
    lines.append("五个指标先各自在测试集模型之间计算名次：R2 越高名次越好，MAE/RMSE 越低名次越好，Acc≤0.5/Acc≤1.0 越高名次越好。并列值使用平均名次。\n\n")
    lines.append("$$Score_5=\\frac{rank_{R^2}+rank_{MAE}+rank_{RMSE}+rank_{Acc\\leq0.5}+rank_{Acc\\leq1.0}}{5}$$\n\n")
    lines.append("Score5 越低越好。该分析用于检验“加入临床容差准确率后，最佳模型是否变化”，不替代正式三指标结论。\n\n")
    lines.append("## 统一解释口径\n\n")
    lines.append("每个性别只解释五指标排名选出的一个最佳模型。该模型的 SHAP 在全部测试样本上计算一次，随后 SHAP summary、SHAP bar、样本热图、waterfall、dependence/interaction、特征排名与 Top30 筛选全部复用这一次结果。\n\n")
    lines.append("$$I_j=\frac{1}{N_{test}}\sum_{i=1}^{N_{test}}|\phi_{ij}|$$\n\n")
    lines.append("Top30 是 `*_shap_feature_rank.csv` 的前 30 行，不再使用 permutation importance、模型内置重要性或跨模型平均重要性。Top30 Pearson 相关在该性别全部有效样本的原始特征值上计算：特征筛选口径是最佳模型 SHAP，相关系数口径是原始数据。辅助阈值分类仅保留在 CSV 指标中，不生成混淆矩阵或其他辅助分类图。\n\n")

    lines.append("## 结果概览\n\n")
    lines.append("| 数据 | 正式三指标最佳 | 敏感性五指标最佳 | Test R2 | MAE | RMSE | Acc≤0.5 | Acc≤1.0 |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|\n")
    for result in sensitivity_results:
        row = result["best_row"]
        lines.append(
            f"| {result['sex']} | {result['official_best_model']} | **{result['best_model_name']}** | "
            f"{report_float(row['R2'])} | {report_float(row['MAE'])} | {report_float(row['RMSE'])} | "
            f"{report_float(row['Acc@0.5score'])} | {report_float(row['Acc@1.0score'])} |\n"
        )

    for index, result in enumerate(sensitivity_results, start=1):
        sex_label = result["sex"]
        lines.append(f"\n## {index + 2}. {sex_label}\n\n")
        lines.append(f"- 正式三指标最佳：`{result['official_best_model']}`。\n")
        lines.append(f"- 五指标敏感性最佳：`{result['best_model_name']}`。\n")
        lines.append(f"- 样本：训练 {result['train_samples']}、测试 {result['test_samples']}；输入特征 {result['input_features']}。\n\n")
        lines.append("| 五指标排名 | 模型 | Score5↓ | R2↑ | MAE↓ | RMSE↓ | Acc≤0.5↑ | Acc≤1.0↑ |\n")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        test_metrics = result["metrics_df"][result["metrics_df"]["split"] == "test"]
        test_metrics = test_metrics.sort_values("model_selection_rank")
        for _, row in test_metrics.iterrows():
            lines.append(
                f"| {int(row['model_selection_rank'])} | {row['model']} | {report_float(row['model_selection_score'])} | "
                f"{report_float(row['R2'])} | {report_float(row['MAE'])} | {report_float(row['RMSE'])} | "
                f"{report_float(row['Acc@0.5score'])} | {report_float(row['Acc@1.0score'])} |\n"
            )
        lines.append("\n关键文件：\n\n")
        lines.append(f"- 五指标排名：`{sex_label}/metrics/model_selection_ranking.csv`。\n")
        lines.append(f"- 三指标与五指标对照：`{sex_label}/metrics/ranking_comparison_3_vs_5_metrics.csv`。\n")
        lines.append(f"- 完整指标和模型：`{sex_label}/metrics/`、`{sex_label}/models/`。\n")
        lines.append(f"- 全模型回归比较图：`{sex_label}/analysis/01_model_comparison/`。\n")
        lines.append(f"- 五指标最佳模型回归图：`{sex_label}/analysis/02_best_model/01_performance/`。\n")
        lines.append(f"- 唯一 SHAP 解释及排名：`{sex_label}/analysis/02_best_model/02_shap/`。\n")
        lines.append(f"- 由同一 SHAP 排名筛出的 Top30 相关与聚类：`{sex_label}/analysis/02_best_model/03_top30_correlation/`。\n")

    lines.append("\n## 输出格式\n\n")
    lines.append("除批量 SHAP interaction 图仅保留 PNG 外，其余图均同时提供 PNG 和 PDF。female/male 子目录是完整副本，可独立读取；本报告位于敏感性分析根目录，性别子目录内不放报告。\n")
    SENSITIVITY_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = SENSITIVITY_ROOT / "report.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"[Sensitivity Report] {report_path}")


def regenerate_sensitivity_analysis_from_existing_outputs():
    results = [
        build_sensitivity_analysis_for_dataset(sex_label, dataset_csv)
        for sex_label, dataset_csv in DATASET_RUNS
    ]
    write_sensitivity_report(results)
    configure_run_paths(DATASET_RUNS[0][1], OUTPUT_ROOT / DATASET_RUNS[0][0])
    return results


def write_combined_report(run_results):
    lines = []
    lines.append("# 女性 / 男性美观评分回归：随机划分训练与可视化报告\n\n")
    lines.append("本报告由同一次脚本运行生成。女性与男性数据完全分开训练、调参、评估和解释；两者都只使用对应的原始 CSV，不混合样本，也不使用 supplementary 数据。\n\n")

    lines.append("## 1. 数据、划分与预处理方法\n\n")
    lines.append(f"- 女性输入：`{DATASET_RUNS[0][1]}`，输出目录：`female/`。\n")
    lines.append(f"- 男性输入：`{DATASET_RUNS[1][1]}`，输出目录：`male/`。\n")
    lines.append(f"- 每套数据独立随机划分：`test_size={base.TEST_SIZE}`、`random_state={base.RANDOM_STATE}`、`shuffle=True`。\n")
    lines.append("- 删除任一输入 measurement 或 `beauty_score` 缺失的样本；输入特征为全部 `measurement__*` 列。\n")
    lines.append("- StandardScaler 仅在各自训练集上拟合，再应用到对应测试集，避免测试集信息进入缩放参数。\n\n")
    lines.append("对任意输入特征或目标，标准化公式为：\n\n")
    lines.append("$$z=\\frac{x-\\mu_{train}}{\\sigma_{train}}$$\n\n")
    lines.append("模型在标准化空间训练；预测评分在计算指标前通过 $x=z\\sigma_{train}+\\mu_{train}$ 还原到原始评分单位。\n\n")

    lines.append("## 2. 评估指标和最优模型选择公式\n\n")
    lines.append("设测试样本数为 $N$，真实评分为 $y_i$，预测评分为 $\\hat y_i$。\n\n")
    lines.append("$$MAE=\\frac{1}{N}\\sum_{i=1}^{N}|y_i-\\hat y_i|$$\n\n")
    lines.append("$$RMSE=\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(y_i-\\hat y_i)^2}$$\n\n")
    lines.append("$$R^2=1-\\frac{\\sum_i(y_i-\\hat y_i)^2}{\\sum_i(y_i-\\bar y)^2}$$\n\n")
    lines.append("MAE、RMSE 越低越好；RMSE 对少量大误差更敏感。$R^2$ 越高越好，测试集上可以小于 0，表示模型还不如直接预测测试集均值。\n\n")
    lines.append("绝对误差准确率为：\n\n")
    lines.append("$$Acc@\\delta=\\frac{1}{N}\\sum_i\\mathbf{1}(|\\hat y_i-y_i|\\leq\\delta),\\quad \\delta\\in\\{0.5,1.0\\}$$\n\n")
    lines.append("相对误差准确率使用 $e_i=|\\hat y_i-y_i|/\\max(|y_i|,10^{-8})$，报告 $e_i\\leq5\\%$ 与 $e_i\\leq10\\%$ 的样本比例。\n\n")
    lines.append(f"辅助分类把真实和预测评分分别按 `{base.BEAUTIFUL_THRESHOLD:g}` 分二类；Accuracy、Precision、Recall、F1 只作为 CSV/报告中的补充数值，不参与回归模型训练，也不生成混淆矩阵或其他辅助分类图。$F1=2PR/(P+R)$。\n\n")
    lines.append("最终模型不是只按一个指标选择。每个模型分别获得 $R^2$ 降序名次、MAE 升序名次和 RMSE 升序名次，然后计算：\n\n")
    lines.append("$$SelectionScore=\\frac{rank_{R^2}+rank_{MAE}+rank_{RMSE}}{3}$$\n\n")
    lines.append("SelectionScore 越小越好；完全并列使用平均名次。若综合分相同，再依次按更高 $R^2$、更低 MAE、更低 RMSE 决定。\n\n")

    lines.append("## 3. 特征解释与相关聚类方法\n\n")
    lines.append("每个性别只对正式三指标选出的最佳模型做一套解释，并在全部测试样本上只计算一次 SHAP：线性模型使用 LinearExplainer，支持的树模型使用 TreeExplainer，其余模型（如 SVR、KNN）使用 KernelExplainer。SHAP 满足：\n\n")
    lines.append("$$f(x_i)=E[f(X)]+\\sum_j\\phi_{ij}$$\n\n")
    lines.append("特征 $j$ 的总体重要性使用全部测试样本的平均绝对 SHAP：\n\n")
    lines.append("$$I_j=\\frac{1}{N_{test}}\\sum_{i=1}^{N_{test}}|\\phi_{ij}|$$\n\n")
    lines.append("`*_shap_feature_rank.csv` 是唯一正式特征排名。SHAP summary、SHAP bar、样本热图、waterfall、dependence/interaction 都复用同一批 SHAP 值；不再计算跨模型平均重要性，也不再把 permutation importance、模型内置重要性或系数图混入正式解释。\n\n")
    lines.append("Top30 相关图严格取上述 SHAP 排名表前 30 行，因此 Top30 名单与 SHAP bar/rank 的前 30 名完全一致。随后在该性别全部有效样本（训练集与测试集合并）的原始特征值上计算 Pearson 相关：\n\n")
    lines.append("$$r_{ij}=\\frac{\\sum_n(x_{ni}-\\bar x_i)(x_{nj}-\\bar x_j)}{\\sqrt{\\sum_n(x_{ni}-\\bar x_i)^2}\\sqrt{\\sum_n(x_{nj}-\\bar x_j)^2}}$$\n\n")
    lines.append("聚类把相关矩阵第 $i$ 行视为特征的相关模式 $p_i=(r_{i1},...,r_{im})$，采用欧氏距离和 average linkage：\n\n")
    lines.append("$$d(i,j)=\\sqrt{\\sum_k(r_{ik}-r_{jk})^2},\\qquad D(A,B)=\\frac{1}{|A||B|}\\sum_{i\\in A}\\sum_{j\\in B}d(i,j)$$\n\n")
    lines.append("因此同一簇表示“对全部 Top30 特征的相关模式相似”，并不代表因果关系，也不等同于两个特征之间的直接 Pearson 相关一定最高。clustermap 会按树状聚类重排显示顺序，但 `top30_overall_features.csv` 保留 SHAP 原始排名顺序。\n\n")

    lines.append("## 4. 可视化阅读说明\n\n")
    lines.append("| 图表 | 计算内容与阅读方法 |\n")
    lines.append("|---|---|\n")
    lines.append("| Test R2 / MAE 柱状图 | 比较各模型测试集解释度与平均绝对误差。R2 越高、MAE 越低越好。 |\n")
    lines.append("| MAE/RMSE 与 R2/Accuracy 曲线 | 模型按综合名次排列，用于观察不同指标是否给出一致结论。 |\n")
    lines.append("| 多指标 grouped bar / heatmap | 展示原始指标值；不同指标量纲不同，应在同一指标内横向比较，不应直接用柱高跨指标判断。 |\n")
    lines.append("| Train/Test R2 gap | $gap=R^2_{train}-R^2_{test}$；较大的正差值提示泛化落差，但应结合测试样本量和测试 R2 一起解释。 |\n")
    lines.append("| 绝对误差箱线图 | 箱体为四分位范围，中线为中位数；叠加散点展示每个测试样本误差。 |\n")
    lines.append("| 有符号误差小提琴图 | 误差定义为 `pred-true`；大于 0 表示高估，小于 0 表示低估，宽度表示密度。 |\n")
    lines.append("| True vs Predicted scatter | 红色 $y=x$ 为理想预测；点越贴近对角线越好。 |\n")
    lines.append("| SHAP summary | 横轴为 SHAP 贡献方向和大小，颜色为特征值高低；同一特征两侧分布反映非线性或个体差异。 |\n")
    lines.append("| SHAP bar / rank | 使用 mean $|SHAP|$ 排序，只表示贡献强度，不直接表示正负方向。 |\n")
    lines.append("| SHAP heatmap | 行为测试解释样本、列为重要特征、颜色为该样本上的 SHAP 正负贡献。 |\n")
    lines.append("| SHAP waterfall | 从 expected value 出发逐项累加单样本 SHAP，展示为何得到该样本预测。 |\n")
    lines.append("| SHAP interaction/dependence | 横轴为特征值、纵轴为该特征 SHAP，颜色为另一特征；用于观察非线性及条件关系，不证明因果。 |\n")
    lines.append("| Top30 correlation heatmap / clustermap | 色块为 Pearson $r$；树枝为相关模式的层次聚类距离。 |\n\n")

    lines.append("## 5. 女性与男性结果\n\n")
    for result in run_results:
        sex_label = result["sex"]
        sex_cn = "女性" if sex_label == "female" else "男性"
        info = result["dataset_info"]
        metrics_df = result["metrics_df"]
        test_metrics = metrics_df[metrics_df["split"] == "test"].sort_values("model_selection_rank")
        train_metrics = metrics_df[metrics_df["split"] == "train"]
        best_model = result["best_model_name"]
        best_test = test_metrics[test_metrics["model"] == best_model].iloc[0]
        best_train = train_metrics[train_metrics["model"] == best_model].iloc[0]
        r2_gap = float(best_train["R2"] - best_test["R2"])

        lines.append(f"### 5.{1 if sex_label == 'female' else 2} {sex_cn}（`{sex_label}/`）\n\n")
        lines.append("| 项目 | 结果 |\n")
        lines.append("|---|---:|\n")
        lines.append(f"| 有效样本数 | {info['total_samples_after_dropna']} |\n")
        lines.append(f"| 训练 / 测试样本 | {info['train_samples']} / {info['test_samples']} |\n")
        lines.append(f"| 输入特征数 | {info['input_features']} |\n")
        lines.append(f"| 最佳模型 | **{best_model}** |\n")
        lines.append(f"| SelectionScore | {report_float(best_test['model_selection_score'])} |\n")
        lines.append(f"| Test R2 / MAE / RMSE | {report_float(best_test['R2'])} / {report_float(best_test['MAE'])} / {report_float(best_test['RMSE'])} |\n")
        lines.append(f"| Acc@0.5 / Acc@1.0 | {report_float(best_test['Acc@0.5score'])} / {report_float(best_test['Acc@1.0score'])} |\n")
        lines.append(f"| Train R2 / Test R2 / gap | {report_float(best_train['R2'])} / {report_float(best_test['R2'])} / {report_float(r2_gap)} |\n\n")

        lines.append("测试集完整模型排名：\n\n")
        lines.append("| 排名 | 模型 | 综合分↓ | MAE↓ | RMSE↓ | R2↑ | Acc≤0.5 | Acc≤1.0 | 辅助Acc | 辅助F1 |\n")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in test_metrics.iterrows():
            lines.append(
                f"| {int(row['model_selection_rank'])} | {row['model']} | {report_float(row['model_selection_score'])} | "
                f"{report_float(row['MAE'])} | {report_float(row['RMSE'])} | {report_float(row['R2'])} | "
                f"{report_float(row['Acc@0.5score'])} | {report_float(row['Acc@1.0score'])} | "
                f"{report_float(row['beautiful_accuracy'])} | {report_float(row['beautiful_f1'])} |\n"
            )

        model_prefix = best_model_file_prefix(best_model)
        importance_path = result["output_dir"] / "analysis" / "02_best_model" / "02_shap" / f"{model_prefix}_shap_feature_rank.csv"
        importance_df = pd.read_csv(importance_path, encoding="utf-8-sig").head(15)
        methods = ", ".join(sorted(importance_df["explain_method"].astype(str).unique()))
        explained_samples = int(importance_df["explained_test_samples"].iloc[0])
        lines.append(f"\n最佳模型 Top15 特征（解释方法：`{methods}`；解释全部 {explained_samples} 个测试样本）：\n\n")
        lines.append("| 排名 | 特征 | mean abs(SHAP) |\n")
        lines.append("|---:|---|---:|\n")
        for _, row in importance_df.iterrows():
            lines.append(f"| {int(row['rank'])} | `{row['feature']}` | {report_float(row['mean_abs_shap'], 6)} |\n")

        lines.append("\n关键文件：\n\n")
        lines.append(f"- 完整指标：`{sex_label}/metrics/metrics_summary.csv`；综合排名：`{sex_label}/metrics/model_selection_ranking.csv`。\n")
        lines.append(f"- 超参数搜索：`{sex_label}/metrics/hyperparameter_search_summary.csv`。\n")
        lines.append(f"- 模型与 scaler：`{sex_label}/models/`；不保存逐样本预测文件。\n")
        lines.append(f"- 全模型回归比较图：`{sex_label}/analysis/01_model_comparison/`。\n")
        lines.append(f"- 最佳模型回归散点：`{sex_label}/analysis/02_best_model/01_performance/`。\n")
        lines.append(f"- 唯一 SHAP 解释、逐样本 SHAP 值与排名：`{sex_label}/analysis/02_best_model/02_shap/`。\n")
        lines.append(f"- Top30 名单、相关矩阵和聚类图：`{sex_label}/analysis/02_best_model/03_top30_correlation/`。\n\n")

    lines.append("## 6. 输出格式约定\n\n")
    lines.append("所有普通图均同时输出 `.png` 和同名 `.pdf`。由于 Top10 SHAP 两两组合会为每个性别产生大量文件，`female/analysis/02_best_model/02_shap/interactions/` 与 `male/.../interactions/` 仅保留 PNG。CSV 保存指标、SHAP 值、SHAP 排名与相关矩阵，便于重新制图和统计复核；不保存逐样本预测 CSV。性别子目录内部不生成 `report.md`，本文件是正式三指标分析的唯一报告；五指标敏感性分析在其独立根目录另有一份总报告。\n")

    report_path = OUTPUT_ROOT / "report.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"[Report] 已生成女性/男性合并报告: {report_path}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_results = []
    for sex_label, dataset_csv in DATASET_RUNS:
        run_results.append(run_single_dataset(sex_label, dataset_csv))
        verify_pdf_coverage(OUTPUT_ROOT / sex_label)
    write_combined_report(run_results)
    regenerate_sensitivity_analysis_from_existing_outputs()
    for sex_label, _ in DATASET_RUNS:
        inner_report = OUTPUT_ROOT / sex_label / "report.md"
        if inner_report.exists():
            inner_report.unlink()
    print(f"[All Complete] Female and male outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
