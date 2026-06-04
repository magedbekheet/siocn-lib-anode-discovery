"""Regenerate public report figures for the final no-DVB app policy."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import TARGET, prepare_features, read_sioc_csv  # noqa: E402
from train_discovery_model import make_groups  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "sioc_battery_capacity_clean_updated.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sioc_app_target_models.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STABLE_TARGET = "cycling_reversible_capacity_mah_g"


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def feature_label(name: str) -> str:
    replacements = {
        "A_composition_T": "Composition + T",
        "B_composition_T_time": "Composition + T + time",
        "C_surface_assisted_composition_T": "Composition + T + surface",
        "D_surface_assisted_composition_T_time": "Composition + T + time + surface",
        "A_stable_composition_T_cycle": "Stable: composition + T + cycle",
        "B_stable_composition_T_time_cycle": "Stable: composition + T + time + cycle",
        "C_surface_assisted_stable_composition_T_cycle": "Stable: + surface",
        "D_surface_assisted_stable_composition_T_time_cycle": "Stable: + time + surface",
        "E_stable_plus_first_capacity_diagnostic": "Stable diagnostic + Qrev",
        "F_surface_assisted_stable_plus_first_capacity_diagnostic": "Stable diagnostic + Qrev + surface",
        "A_material_plus_first_reversible_capacity": "CE diagnostic + Qrev",
    }
    return replacements.get(name, name.replace("_", " "))


def load_data() -> pd.DataFrame:
    raw = read_sioc_csv(str(DATA_PATH))
    df = prepare_features(raw)
    df["doi_group"] = make_groups(df).values
    return df


def plot_target_distribution(df: pd.DataFrame) -> None:
    first = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    stable = pd.to_numeric(df[STABLE_TARGET], errors="coerce").dropna() if STABLE_TARGET in df else pd.Series(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(first, bins=24, kde=True, ax=axes[0], color="#236f68")
    axes[0].set_title("First-cycle reversible capacity")
    axes[0].set_xlabel("Qrev (mAh/g)")
    axes[0].set_ylabel("Count")
    if not stable.empty:
        sns.histplot(stable, bins=18, kde=True, ax=axes[1], color="#176b87")
    axes[1].set_title("Cycled reversible capacity")
    axes[1].set_xlabel("Qcycled (mAh/g)")
    axes[1].set_ylabel("Count")
    savefig(FIG_DIR / "target_distribution.png")


def plot_best_feature_sets() -> None:
    best_path = REPORT_DIR / "final_simplified_model_best_by_feature_set.csv"
    best = pd.read_csv(best_path)
    first = best[(best["target"] == TARGET) & (best["cv_kind"] == "grouped")].copy()
    first["feature_label"] = first["feature_set"].map(feature_label)
    first = first.sort_values("mae_mean")

    plt.figure(figsize=(9, 4.6))
    ax = sns.barplot(data=first, x="mae_mean", y="feature_label", hue="model", dodge=False, palette="viridis")
    ax.set_title("Final first-cycle Qrev models: grouped CV")
    ax.set_xlabel("MAE (mAh/g), lower is better")
    ax.set_ylabel("")
    ax.legend(title="Estimator", loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)
    savefig(FIG_DIR / "best_feature_sets_mae.png")


def grouped_oof(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    data = df.dropna(subset=[bundle["target"]]).copy()
    cols = list(bundle["feature_columns"])
    X = data[cols].copy()
    y = data[bundle["target"]].copy()
    groups = data["doi_group"]
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    pred = np.full(len(data), np.nan)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = clone(bundle["model"])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])
    out = data.copy()
    out["oof_pred"] = pred
    return out


def plot_fit_and_importance(df: pd.DataFrame, bundle: dict) -> None:
    out = grouped_oof(df, bundle)
    y = out[bundle["target"]]
    pred = out["oof_pred"]
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)

    plt.figure(figsize=(5.2, 5.0))
    ax = sns.scatterplot(data=out, x=bundle["target"], y="oof_pred", hue=out["n_wt_pct"].fillna(0) > 0, palette={False: "#236f68", True: "#b7791f"}, s=45)
    lo = min(y.min(), pred.min())
    hi = max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#333333", linewidth=1)
    ax.set_title(f"Grouped-CV fit: final Qrev model\nMAE {mae:.1f} mAh/g, R2 {r2:.2f}")
    ax.set_xlabel("Measured Qrev (mAh/g)")
    ax.set_ylabel("Predicted Qrev (mAh/g)")
    ax.legend(title="N-containing", loc="upper left")
    savefig(FIG_DIR / "holdout_predicted_vs_observed_A_design_stage.png")

    reg = bundle["model"].named_steps["regressor"]
    features = list(bundle["feature_columns"])
    importances = getattr(reg, "feature_importances_", None)
    if importances is None:
        importances = np.abs(getattr(reg, "coef_", np.zeros(len(features))))
    imp = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    plt.figure(figsize=(7, 4.2))
    ax = sns.barplot(data=imp, x="importance", y="feature", color="#236f68")
    ax.set_title("Final Qrev model feature importance")
    ax.set_xlabel("Model importance")
    ax.set_ylabel("")
    savefig(FIG_DIR / "tree_importance_A_design_stage.png")

    try:
        import shap

        X = df.dropna(subset=[bundle["target"]])[features].copy()
        X = X.sample(min(len(X), 200), random_state=42)
        transformed = bundle["model"].named_steps["preprocessor"].transform(X)
        explainer = shap.TreeExplainer(reg)
        shap_values = explainer.shap_values(transformed)
        shap.summary_plot(
            shap_values,
            transformed,
            feature_names=features,
            show=False,
            max_display=len(features),
            plot_size=(7.2, 4.6),
        )
        plt.title("SHAP summary: final Qrev design model", fontsize=12)
        savefig(FIG_DIR / "shap_summary_A_design_stage.png")
    except Exception as exc:  # pragma: no cover - plot fallback for lightweight environments
        print(f"SHAP summary failed ({exc}); writing model-importance fallback.")
        plt.figure(figsize=(7, 4.2))
        ax = sns.barplot(data=imp, x="importance", y="feature", color="#176b87")
        ax.set_title("Final Qrev explanation summary\n(model-importance fallback)")
        ax.set_xlabel("Model importance")
        ax.set_ylabel("")
        savefig(FIG_DIR / "shap_summary_A_design_stage.png")

    out = out.copy()
    out["residual"] = out["oof_pred"] - out[bundle["target"]]
    plt.figure(figsize=(7.2, 4.2))
    ax = sns.histplot(out["residual"], bins=24, kde=True, color="#236f68")
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Grouped-CV residuals: final Qrev model")
    ax.set_xlabel("Prediction residual, predicted - measured (mAh/g)")
    ax.set_ylabel("Count")
    savefig(FIG_DIR / "residual_distributions.png")


def plot_leakage_correlations(df: pd.DataFrame) -> None:
    cols = [
        TARGET,
        "irreversible_capacity_mah_g",
        "coulombic_efficiency_pct",
        STABLE_TARGET,
        "capacity_retention_pct",
        "cycling_numbers",
        "si_wt_pct",
        "c_wt_pct",
        "o_wt_pct",
        "n_wt_pct",
        "pyrolysis_temp_c",
        "pyrolysis_time_h",
        "surface_area_m2_g",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(method="spearman", numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.3)
    plt.title("Spearman correlations: targets, diagnostics, and final descriptors")
    savefig(FIG_DIR / "leakage_proxy_correlations.png")


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_data()
    bundles = joblib.load(MODEL_PATH)
    qrev_bundle = bundles["first_reversible"]
    plot_target_distribution(df)
    plot_best_feature_sets()
    plot_fit_and_importance(df, qrev_bundle)
    plot_leakage_correlations(df)


if __name__ == "__main__":
    main()
