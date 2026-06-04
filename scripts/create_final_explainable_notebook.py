"""Create an explainable final SiOC/SiOCN modeling notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip() + "\n",
    }


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def main() -> None:
    nb = {
        "cells": [],
        "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    cells = [
        md(
            """
            # Final Explainable SiOC/SiOCN Anode Modeling

            This notebook is the final cleaned modeling workflow for the Si-O-C / Si-O-C-N lithium-ion battery anode dataset.

            The project goal is to support **composition and processing discovery** for stable anode materials. The notebook therefore separates:

            - a **primary discovery model** for first reported reversible capacity using only pre-test design-stage features,
            - a **stable/cycled-capacity model** using cycling-capacity data where available,
            - diagnostic/interpolation tests such as shuffled CV, first-capacity-assisted stable prediction, ANN/MLP benchmarks, PCA, feature importance, and SHAP.

            The main reported metric remains DOI/reference-grouped cross-validation, because this estimates transfer to unseen literature families rather than interpolation within the same paper.
            """
        ),
        md("## 1. Imports and Paths"),
        code(
            """
            from __future__ import annotations

            from pathlib import Path
            import sys
            import warnings

            import joblib
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            import shap

            from sklearn.base import clone
            from sklearn.compose import ColumnTransformer
            from sklearn.dummy import DummyRegressor
            from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.inspection import permutation_importance
            from sklearn.linear_model import ElasticNet, Ridge
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import GroupKFold, KFold, cross_validate
            from sklearn.neural_network import MLPRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.svm import SVR
            from sklearn.decomposition import PCA

            warnings.filterwarnings("ignore")
            sns.set_theme(style="whitegrid", context="notebook")
            RANDOM_STATE = 42

            cwd = Path.cwd().resolve()
            PROJECT_ROOT = next(
                (root for root in [cwd, cwd.parent, cwd.parent.parent] if (root / "data" / "sioc_battery_capacity_clean_updated.csv").exists()),
                None,
            )
            if PROJECT_ROOT is None:
                raise FileNotFoundError("Could not find project root.")
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

	            DATA_PATH = PROJECT_ROOT / "data" / "sioc_battery_capacity_clean_updated.csv"
	            MODEL_DIR = PROJECT_ROOT / "models"
	            REPORT_DIR = PROJECT_ROOT / "reports"
	            FIG_DIR = REPORT_DIR / "figures"
	            MODEL_DIR.mkdir(parents=True, exist_ok=True)
	            REPORT_DIR.mkdir(parents=True, exist_ok=True)
	            FIG_DIR.mkdir(parents=True, exist_ok=True)

            print("Project root:", PROJECT_ROOT)
            print("Data path:", DATA_PATH)
            """
        ),
        md("## 2. Load Data and Shared Feature Engineering"),
        code(
            """
            from src.features import (
                read_sioc_csv,
                standardize_columns,
                prepare_features,
                TARGET,
                PRIMARY_DISCOVERY_NUMERIC_FEATURES,
                PROTOCOL_CONDITIONING_FEATURES,
                PRIMARY_DISCOVERY_CATEGORICAL_FEATURES,
                OPTIONAL_STRUCTURE_FEATURES,
                PHASE_PHYSICS_FEATURES,
            )
            from train_discovery_model import make_groups

            STABLE_TARGET = "cycling_reversible_capacity_mah_g"

	            raw = read_sioc_csv(str(DATA_PATH))
	            std = standardize_columns(raw)
	            df = prepare_features(raw)
	            df_first = df.dropna(subset=[TARGET]).copy()
	            df_first["doi_group"] = make_groups(df_first).values

            df_stable = df.dropna(subset=[STABLE_TARGET]).copy()
            df_stable = df_stable.dropna(subset=["cycling_numbers"], how="all").copy()
            df_stable["doi_group"] = make_groups(df_stable).values

            print("Raw shape:", raw.shape)
            print("First/reversible target rows:", len(df_first), "groups:", df_first["doi_group"].nunique())
            print("Stable/cycled target rows:", len(df_stable), "groups:", df_stable["doi_group"].nunique())
            display(df_first.head())
            """
        ),
        md(
            """
            ## 3. Dataset and Target Audit

            The reversible-capacity target is more complete, while the stable/cycled target is smaller and more heterogeneous. N-containing samples are sparse, so N-doping trends should be interpreted cautiously.
            """
        ),
        code(
            """
            audit = pd.DataFrame([
                {"item": "Rows in cleaned dataset", "value": len(df_first)},
                {"item": "Rows with stable/cycled capacity", "value": len(df_stable)},
                {"item": "DOI/reference groups", "value": df_first["doi_group"].nunique()},
                {"item": "N-containing rows", "value": int((df_first["n_wt_pct"].fillna(0) > 0).sum())},
                {"item": "Rows with 0-3 V max", "value": int((pd.to_numeric(df_first["voltage_max_v"], errors="coerce") == 3.0).sum())},
                {"item": "Rows with 18-18.6 mA/g first current", "value": int(pd.to_numeric(df_first["first_cycling_current_ma_g"], errors="coerce").between(18.0, 18.6).sum())},
            ])
            display(audit)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            sns.histplot(df_first[TARGET], bins=24, kde=True, ax=axes[0])
            axes[0].set_title("First reported reversible capacity")
            axes[0].set_xlabel("mAh/g")

            sns.histplot(df_stable[STABLE_TARGET], bins=18, kde=True, ax=axes[1], color="tab:green")
            axes[1].set_title("Stable/cycled capacity subset")
            axes[1].set_xlabel("mAh/g")

            sns.histplot(df_first["n_wt_pct"].fillna(0), bins=20, ax=axes[2], color="tab:orange")
            axes[2].set_title("N wt.% distribution")
            axes[2].set_xlabel("N wt.%")
            plt.tight_layout()
            plt.show()

            display(df_first["polymer_family_broad"].value_counts(dropna=False).rename_axis("polymer_family_broad").reset_index(name="count"))
            """
        ),
        md("## 4. Feature Policy"),
        code(
            """
            COMPOSITION_FEATURES = ["si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct"]
            PYROLYSIS_TEMP_FEATURES = ["pyrolysis_temp_c"]
            PYROLYSIS_TIME_FEATURES = ["pyrolysis_time_h"]
            PROCESSING_FEATURES = PYROLYSIS_TEMP_FEATURES + PYROLYSIS_TIME_FEATURES
            FIRST_PROTOCOL_FEATURES = []
            CYCLING_PROTOCOL_FEATURES = ["cycling_numbers"]
            BASE_CATEGORICAL = []
            ATMOSPHERE_FEATURES = ["pyrolysis_atmosphere_group"]
            STRUCTURE_FEATURES = ["surface_area_m2_g"]
            PHASE_FEATURES = PHASE_PHYSICS_FEATURES
            DVB_DIAGNOSTIC_FEATURES = ["dvb_ratio_to_base", "dvb_modification"]

            FIRST_DESIGN_FEATURES = COMPOSITION_FEATURES + PROCESSING_FEATURES
            FIRST_FEATURE_SETS = {
                "A_composition_T": COMPOSITION_FEATURES + PYROLYSIS_TEMP_FEATURES,
                "B_composition_T_time": FIRST_DESIGN_FEATURES,
                "C_surface_assisted_composition_T": COMPOSITION_FEATURES + PYROLYSIS_TEMP_FEATURES + STRUCTURE_FEATURES,
                "D_surface_assisted_composition_T_time": FIRST_DESIGN_FEATURES + STRUCTURE_FEATURES,
            }
            DEPLOYED_FIRST_FEATURE_SET = "B_composition_T_time"
            SURFACE_FIRST_FEATURE_SET = "D_surface_assisted_composition_T_time"

            STABLE_DESIGN_FEATURES = FIRST_DESIGN_FEATURES + FIRST_PROTOCOL_FEATURES + CYCLING_PROTOCOL_FEATURES
            STABLE_FEATURE_SETS = {
                "A_stable_composition_T_cycle": COMPOSITION_FEATURES + PYROLYSIS_TEMP_FEATURES + CYCLING_PROTOCOL_FEATURES,
                "B_stable_composition_T_time_cycle": STABLE_DESIGN_FEATURES,
                "C_surface_assisted_stable_composition_T_cycle": COMPOSITION_FEATURES + PYROLYSIS_TEMP_FEATURES + CYCLING_PROTOCOL_FEATURES + STRUCTURE_FEATURES,
                "D_surface_assisted_stable_composition_T_time_cycle": STABLE_DESIGN_FEATURES + STRUCTURE_FEATURES,
                "E_stable_plus_first_capacity_diagnostic": STABLE_DESIGN_FEATURES + [TARGET],
                "F_surface_assisted_stable_plus_first_capacity_diagnostic": STABLE_DESIGN_FEATURES + STRUCTURE_FEATURES + [TARGET],
            }
            DEPLOYED_STABLE_FEATURE_SET = "B_stable_composition_T_time_cycle"
            SURFACE_STABLE_FEATURE_SET = "D_surface_assisted_stable_composition_T_time_cycle"
            DEPLOYED_STABLE_DIAGNOSTIC_FEATURE_SET = "E_stable_plus_first_capacity_diagnostic"
            SURFACE_STABLE_DIAGNOSTIC_FEATURE_SET = "F_surface_assisted_stable_plus_first_capacity_diagnostic"

            FORBIDDEN_FIRST_FEATURES = {
                TARGET,
                "irreversible_capacity_mah_g",
                "coulombic_efficiency_pct",
                "cycling_reversible_capacity_mah_g",
                "capacity_retention_pct",
                "cycling_numbers",
                "cycling_current_ma_g",
            }
            FORBIDDEN_STABLE_PRIMARY_FEATURES = {
                STABLE_TARGET,
                "capacity_retention_pct",
                "irreversible_capacity_mah_g",
                "coulombic_efficiency_pct",
            }

            for name, cols in FIRST_FEATURE_SETS.items():
                FIRST_FEATURE_SETS[name] = [c for c in cols if c in df_first.columns]
                forbidden = sorted(set(FIRST_FEATURE_SETS[name]).intersection(FORBIDDEN_FIRST_FEATURES))
                if forbidden:
                    raise ValueError(f"Leaky first-capacity features in {name}: {forbidden}")

            for name, cols in STABLE_FEATURE_SETS.items():
                STABLE_FEATURE_SETS[name] = [c for c in cols if c in df_stable.columns]
                if "first_capacity_diagnostic" not in name:
                    forbidden = sorted(set(STABLE_FEATURE_SETS[name]).intersection(FORBIDDEN_STABLE_PRIMARY_FEATURES | {TARGET}))
                    if forbidden:
                        raise ValueError(f"Leaky stable-capacity features in {name}: {forbidden}")

            display(pd.DataFrame([
                {"target": "first_reversible", "feature_set": k, "n_features": len(v), "features": ", ".join(v)}
                for k, v in FIRST_FEATURE_SETS.items()
            ] + [
                {"target": "stable_cycled", "feature_set": k, "n_features": len(v), "features": ", ".join(v)}
                for k, v in STABLE_FEATURE_SETS.items()
            ]))
            """
        ),
        md("## 5. Modeling Utilities"),
        code(
            """
            def split_feature_types(cols, data):
                numeric_cols, categorical_cols = [], []
                for col in cols:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        numeric_cols.append(col)
                    else:
                        categorical_cols.append(col)
                return numeric_cols, categorical_cols

            def make_preprocessor(cols, data):
                numeric_cols, categorical_cols = split_feature_types(cols, data)
                transformers = []
                if numeric_cols:
                    transformers.append(("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]), numeric_cols))
                if categorical_cols:
                    transformers.append(("cat", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]), categorical_cols))
                return ColumnTransformer(transformers, remainder="drop")

            def make_model(model_name, cols, data):
                models = {
                    "Baseline mean": DummyRegressor(strategy="mean"),
                    "Ridge": Ridge(alpha=10.0),
                    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=20000),
                    "SVR RBF": SVR(C=10.0, gamma="scale", epsilon=20.0),
                    "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=5, min_samples_leaf=5, min_samples_split=10, random_state=RANDOM_STATE, n_jobs=1),
                    "Extra Trees": ExtraTreesRegressor(n_estimators=400, max_depth=5, min_samples_leaf=5, min_samples_split=10, random_state=RANDOM_STATE, n_jobs=1),
                    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.035, max_depth=2, min_samples_leaf=6, subsample=0.85, random_state=RANDOM_STATE),
                    "ANN small MLP": MLPRegressor(hidden_layer_sizes=(16,), activation="relu", solver="adam", alpha=5.0, learning_rate_init=0.003, max_iter=2500, early_stopping=True, validation_fraction=0.2, n_iter_no_change=60, random_state=RANDOM_STATE),
                }
                return Pipeline([("preprocessor", make_preprocessor(cols, data)), ("regressor", models[model_name])])

            CORE_MODELS = ["Baseline mean", "Ridge", "ElasticNet", "SVR RBF", "Random Forest", "Extra Trees", "Gradient Boosting"]
            ANN_BENCHMARK_MODELS = ["Gradient Boosting", "ANN small MLP"]

            def cv_object(data, cv_kind):
                if cv_kind == "group_doi_or_reference":
                    groups = data["doi_group"]
                    return GroupKFold(n_splits=min(5, groups.nunique())), groups
                if cv_kind == "shuffled_kfold_no_doi_grouping":
                    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), None
                raise ValueError(cv_kind)

            def evaluate_feature_sets(data, target, feature_sets, cv_kind, model_names=CORE_MODELS):
                cv, groups = cv_object(data, cv_kind)
                rows = []
                for fs_name, cols in feature_sets.items():
                    X = data[cols].copy()
                    y = data[target].copy()
                    for model_name in model_names:
                        pipe = make_model(model_name, cols, data)
                        scores = cross_validate(
                            pipe, X, y, cv=cv, groups=groups,
                            scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
                            return_train_score=True, n_jobs=1, error_score="raise",
                        )
                        rows.append({
                            "target": target,
                            "feature_set": fs_name,
                            "cv_kind": cv_kind,
                            "model": model_name,
                            "n_rows": len(data),
                            "n_groups": data["doi_group"].nunique(),
                            "n_features_raw": len(cols),
                            "train_r2_mean": scores["train_r2"].mean(),
                            "test_r2_mean": scores["test_r2"].mean(),
                            "test_r2_std": scores["test_r2"].std(),
                            "mae_mean": -scores["test_mae"].mean(),
                            "mae_std": scores["test_mae"].std(),
                            "rmse_mean": -scores["test_rmse"].mean(),
                            "rmse_std": scores["test_rmse"].std(),
                        })
                return pd.DataFrame(rows)

            def best_by_feature_set(results):
                return (
                    results[results["model"] != "Baseline mean"]
                    .sort_values(["cv_kind", "feature_set", "mae_mean", "rmse_mean"])
                    .groupby(["cv_kind", "feature_set"], as_index=False)
                    .first()
                    .sort_values(["cv_kind", "mae_mean"])
                )

            def grouped_oof_predictions(data, target, cols, model_name="Gradient Boosting", cv_kind="group_doi_or_reference"):
                cv, groups = cv_object(data, cv_kind)
                X = data[cols].copy()
                y = data[target].copy()
                pred = np.full(len(data), np.nan)
                fold = np.full(len(data), -1)
                for i, (tr, te) in enumerate(cv.split(X, y, groups)):
                    model = make_model(model_name, cols, data)
                    model.fit(X.iloc[tr], y.iloc[tr])
                    pred[te] = model.predict(X.iloc[te])
                    fold[te] = i
                out = data.copy()
                out["oof_pred"] = pred
                out["abs_error"] = (out[target] - out["oof_pred"]).abs()
                out["signed_error"] = out["oof_pred"] - out[target]
                out["fold"] = fold
                return out
            """
	        ),
	        md(
	            """
	            ### Why Two CV Scenarios Are Reported

	            This notebook reports two validation views:

	            - **DOI/reference-grouped CV**: all rows from the same paper/source stay together in either train or test. This is the primary metric because it tests transfer to an unseen literature work.
	            - **Shuffled CV**: rows are randomly split. This can put neighboring samples from the same DOI/source in both train and test. It is useful as an optimistic interpolation diagnostic, but it is not used for deployed model selection.

	            The final app and README therefore privilege grouped CV. Shuffled CV is kept because the gap between grouped and shuffled results is scientifically informative: it shows how much performance comes from interpolation inside known paper/precursor families.
	            """
	        ),
	        md("## 6. First/Reversible Capacity Model Comparison"),
        code(
            """
            first_results = pd.concat([
                evaluate_feature_sets(df_first, TARGET, FIRST_FEATURE_SETS, "group_doi_or_reference", CORE_MODELS),
                evaluate_feature_sets(df_first, TARGET, FIRST_FEATURE_SETS, "shuffled_kfold_no_doi_grouping", CORE_MODELS),
            ], ignore_index=True)
            first_best = best_by_feature_set(first_results)
            display(first_best[["cv_kind", "feature_set", "model", "test_r2_mean", "mae_mean", "rmse_mean", "n_features_raw"]])

	            first_results.to_csv(MODEL_DIR / "final_explainable_first_capacity_cv_results.csv", index=False)
	            first_best.to_csv(MODEL_DIR / "final_explainable_first_capacity_best_by_feature_set.csv", index=False)
	            """
	        ),
	        code(
	            """
	            first_cv_diagnostic = (
	                first_best[first_best["feature_set"].isin([DEPLOYED_FIRST_FEATURE_SET, SURFACE_FIRST_FEATURE_SET])]
	                [["cv_kind", "feature_set", "model", "mae_mean", "test_r2_mean", "n_rows", "n_groups"]]
	                .sort_values(["feature_set", "cv_kind"])
	            )
	            display(first_cv_diagnostic)
	            """
	        ),
        code(
            """
            plt.figure(figsize=(10, 4))
            plot_df = first_best.copy()
            sns.barplot(data=plot_df, x="feature_set", y="mae_mean", hue="cv_kind")
            plt.ylabel("CV MAE (mAh/g)")
            plt.xlabel("")
            plt.title("First/reversible capacity: grouped vs shuffled CV")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## 7. Stable/Cycled Capacity Model Comparison"),
        code(
            """
            stable_results = pd.concat([
                evaluate_feature_sets(df_stable, STABLE_TARGET, STABLE_FEATURE_SETS, "group_doi_or_reference", CORE_MODELS),
                evaluate_feature_sets(df_stable, STABLE_TARGET, STABLE_FEATURE_SETS, "shuffled_kfold_no_doi_grouping", CORE_MODELS),
            ], ignore_index=True)
            stable_best = best_by_feature_set(stable_results)
            display(stable_best[["cv_kind", "feature_set", "model", "test_r2_mean", "mae_mean", "rmse_mean", "n_features_raw"]])

	            stable_results.to_csv(MODEL_DIR / "final_explainable_stable_capacity_cv_results.csv", index=False)
	            stable_best.to_csv(MODEL_DIR / "final_explainable_stable_capacity_best_by_feature_set.csv", index=False)
	            """
	        ),
	        code(
	            """
	            stable_cv_diagnostic = (
	                stable_best[stable_best["feature_set"].isin([
	                    DEPLOYED_STABLE_FEATURE_SET,
	                    DEPLOYED_STABLE_DIAGNOSTIC_FEATURE_SET,
	                    SURFACE_STABLE_DIAGNOSTIC_FEATURE_SET,
	                ])]
	                [["cv_kind", "feature_set", "model", "mae_mean", "test_r2_mean", "n_rows", "n_groups"]]
	                .sort_values(["feature_set", "cv_kind"])
	            )
	            display(stable_cv_diagnostic)
	            """
	        ),
        code(
            """
            plt.figure(figsize=(11, 4))
            sns.barplot(data=stable_best, x="feature_set", y="mae_mean", hue="cv_kind")
            plt.ylabel("CV MAE (mAh/g)")
            plt.xlabel("")
            plt.title("Stable/cycled capacity: design model vs diagnostics")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 8. ANN/MLP Benchmark

            ANN models are included as a sanity check because neural networks are often suggested for materials prediction. In this small-data setting they are expected to be less robust than tree ensembles.
            """
        ),
        code(
            """
            ann_first = evaluate_feature_sets(
                df_first, TARGET, {DEPLOYED_FIRST_FEATURE_SET: FIRST_FEATURE_SETS[DEPLOYED_FIRST_FEATURE_SET]},
                "group_doi_or_reference", ANN_BENCHMARK_MODELS,
            )
            ann_first_shuffled = evaluate_feature_sets(
                df_first, TARGET, {DEPLOYED_FIRST_FEATURE_SET: FIRST_FEATURE_SETS[DEPLOYED_FIRST_FEATURE_SET]},
                "shuffled_kfold_no_doi_grouping", ANN_BENCHMARK_MODELS,
            )
            ann_stable = evaluate_feature_sets(
                df_stable, STABLE_TARGET, {DEPLOYED_STABLE_FEATURE_SET: STABLE_FEATURE_SETS[DEPLOYED_STABLE_FEATURE_SET]},
                "group_doi_or_reference", ANN_BENCHMARK_MODELS,
            )
            ann_results = pd.concat([ann_first, ann_first_shuffled, ann_stable], ignore_index=True)
            ann_results.to_csv(MODEL_DIR / "final_explainable_ann_benchmark_results.csv", index=False)
            display(ann_results[["target", "cv_kind", "feature_set", "model", "test_r2_mean", "mae_mean", "rmse_mean"]].sort_values(["target", "cv_kind", "mae_mean"]))
            """
        ),
        md("## 9. Out-of-Fold Fits and DOI Group Audit"),
        code(
            """
            primary_model_name = first_best[
                (first_best["cv_kind"] == "group_doi_or_reference")
                & (first_best["feature_set"] == DEPLOYED_FIRST_FEATURE_SET)
            ]["model"].iloc[0]

            first_oof_grouped = grouped_oof_predictions(
                df_first, TARGET, FIRST_FEATURE_SETS[DEPLOYED_FIRST_FEATURE_SET], primary_model_name, "group_doi_or_reference"
            )
            first_oof_shuffled = grouped_oof_predictions(
                df_first, TARGET, FIRST_FEATURE_SETS[DEPLOYED_FIRST_FEATURE_SET], primary_model_name, "shuffled_kfold_no_doi_grouping"
            )

            print("Grouped OOF MAE:", mean_absolute_error(first_oof_grouped[TARGET], first_oof_grouped["oof_pred"]))
            print("Grouped OOF R2:", r2_score(first_oof_grouped[TARGET], first_oof_grouped["oof_pred"]))
            print("Shuffled OOF MAE:", mean_absolute_error(first_oof_shuffled[TARGET], first_oof_shuffled["oof_pred"]))
            print("Shuffled OOF R2:", r2_score(first_oof_shuffled[TARGET], first_oof_shuffled["oof_pred"]))

            fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
            for ax, data, title in [
                (axes[0], first_oof_grouped, "DOI/reference-grouped CV"),
                (axes[1], first_oof_shuffled, "Shuffled CV"),
            ]:
                sns.scatterplot(data=data, x=TARGET, y="oof_pred", hue=(data["n_wt_pct"].fillna(0) > 0), ax=ax, palette={False: "tab:blue", True: "tab:orange"}, legend=False)
                lo = min(data[TARGET].min(), data["oof_pred"].min())
                hi = max(data[TARGET].max(), data["oof_pred"].max())
                ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
                ax.set_title(title)
                ax.set_xlabel("Measured capacity (mAh/g)")
                ax.set_ylabel("Predicted capacity (mAh/g)")
            plt.tight_layout()
            plt.show()

            group_audit = first_oof_grouped.groupby(["reference", "doi", "doi_group"], dropna=False).agg(
                n=("id", "size"),
                target_mean=(TARGET, "mean"),
                pred_mean=("oof_pred", "mean"),
                mae=("abs_error", "mean"),
                max_error=("abs_error", "max"),
                target_min=(TARGET, "min"),
                target_max=(TARGET, "max"),
            ).reset_index()
            group_audit["target_range"] = group_audit["target_max"] - group_audit["target_min"]
            group_audit = group_audit.sort_values(["mae", "target_range"], ascending=False)
            group_audit.to_csv(REPORT_DIR / "final_explainable_doi_group_audit.csv", index=False)
            display(group_audit.head(15))
            """
        ),
        md("## 10. Correlation and Importance Screening"),
        code(
            """
            corr_cols = [
                TARGET, STABLE_TARGET, "si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct",
                "pyrolysis_temp_c", "pyrolysis_time_h",
                "cycling_numbers",
                "surface_area_m2_g", "log_surface_area_m2_g",
                "conductive_network_index", "ceramic_confinement_index", "effective_bandgap_proxy_ev",
                "phase_weighted_electronegativity", "phase_separation_index",
            ]
            corr_cols = [c for c in corr_cols if c in df.columns]
            corr = df[corr_cols].corr(method="spearman", numeric_only=True)
            plt.figure(figsize=(12, 9))
            sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
            plt.title("Spearman correlation: targets and numeric descriptors")
            plt.tight_layout()
            plt.show()

            target_corr = corr[[TARGET, STABLE_TARGET]].dropna(how="all").sort_values(TARGET, key=lambda s: s.abs(), ascending=False)
            display(target_corr)
            """
        ),
        md("## 11. PCA Map of Design Space"),
        code(
            """
            pca_features = [c for c in FIRST_DESIGN_FEATURES if c not in BASE_CATEGORICAL and c in df_first.columns]
            pca_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=2, random_state=RANDOM_STATE)),
            ])
            pcs = pca_pipe.fit_transform(df_first[pca_features])
            pca_df = df_first[["id", "reference", "doi", TARGET, "n_wt_pct", "polymer_family_broad", "precursor_family"]].copy()
            pca_df["PC1"] = pcs[:, 0]
            pca_df["PC2"] = pcs[:, 1]
            pca_df["N_doped"] = pca_df["n_wt_pct"].fillna(0) > 0
            explained = pca_pipe.named_steps["pca"].explained_variance_ratio_

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=TARGET, style="N_doped", palette="viridis", ax=axes[0])
            axes[0].set_title(f"PCA colored by capacity (PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%})")
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="polymer_family_broad", style="N_doped", ax=axes[1])
            axes[1].set_title("PCA colored by broad polymer family")
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            plt.tight_layout()
            plt.show()

            loadings = pd.DataFrame(
                pca_pipe.named_steps["pca"].components_.T,
                index=pca_features,
                columns=["PC1_loading", "PC2_loading"],
            )
            display(loadings.reindex(loadings["PC1_loading"].abs().sort_values(ascending=False).index))
            """
        ),
        md("## 12. Feature Importance and SHAP for the Final Primary Model"),
        code(
            """
            primary_features = FIRST_FEATURE_SETS[DEPLOYED_FIRST_FEATURE_SET]
            primary_model = make_model(primary_model_name, primary_features, df_first)
            X_primary = df_first[primary_features].copy()
            y_primary = df_first[TARGET].copy()
            primary_model.fit(X_primary, y_primary)

            perm = permutation_importance(
                primary_model, X_primary, y_primary,
                scoring="neg_mean_absolute_error", n_repeats=20, random_state=RANDOM_STATE, n_jobs=1,
            )
            perm_df = pd.DataFrame({
                "feature": primary_features,
                "importance_mae_increase": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mae_increase", ascending=False)
            display(perm_df)
            plt.figure(figsize=(8, 5))
            sns.barplot(data=perm_df.head(15), y="feature", x="importance_mae_increase", color="tab:blue")
            plt.xlabel("Permutation importance: MAE increase")
            plt.ylabel("")
            plt.title("Primary model permutation importance")
            plt.tight_layout()
            plt.show()

            pre = primary_model.named_steps["preprocessor"]
            reg = primary_model.named_steps["regressor"]
            X_trans = pre.transform(X_primary)
            feature_names = pre.get_feature_names_out()
            X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

            explainer = shap.Explainer(reg, X_trans_df)
            shap_values = explainer(X_trans_df, check_additivity=False)
            shap.summary_plot(shap_values, X_trans_df, plot_type="bar", show=False, max_display=20)
            plt.title("SHAP global importance: primary first-capacity model")
            plt.tight_layout()
            plt.show()

            shap.summary_plot(shap_values, X_trans_df, show=False, max_display=20)
            plt.title("SHAP distribution: primary first-capacity model")
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## 13. Train and Save Final Model Bundles"),
        code(
            """
            stable_primary_model_name = stable_best[
                (stable_best["cv_kind"] == "group_doi_or_reference")
                & (stable_best["feature_set"] == DEPLOYED_STABLE_FEATURE_SET)
            ]["model"].iloc[0]
            stable_primary_features = STABLE_FEATURE_SETS[DEPLOYED_STABLE_FEATURE_SET]
            stable_primary_model = make_model(stable_primary_model_name, stable_primary_features, df_stable)
            stable_primary_model.fit(df_stable[stable_primary_features].copy(), df_stable[STABLE_TARGET].copy())

            stable_surface_model_name = stable_best[
                (stable_best["cv_kind"] == "group_doi_or_reference")
                & (stable_best["feature_set"] == SURFACE_STABLE_FEATURE_SET)
            ]["model"].iloc[0]
            stable_surface_features = STABLE_FEATURE_SETS[SURFACE_STABLE_FEATURE_SET]
            stable_surface_model = make_model(stable_surface_model_name, stable_surface_features, df_stable)
            stable_surface_model.fit(df_stable[stable_surface_features].copy(), df_stable[STABLE_TARGET].copy())

            stable_diagnostic_model_name = stable_best[
                (stable_best["cv_kind"] == "group_doi_or_reference")
                & (stable_best["feature_set"] == DEPLOYED_STABLE_DIAGNOSTIC_FEATURE_SET)
            ]["model"].iloc[0]
            stable_diagnostic_features = STABLE_FEATURE_SETS[DEPLOYED_STABLE_DIAGNOSTIC_FEATURE_SET]
            stable_diagnostic_model = make_model(stable_diagnostic_model_name, stable_diagnostic_features, df_stable)
            stable_diagnostic_model.fit(df_stable[stable_diagnostic_features].copy(), df_stable[STABLE_TARGET].copy())

            stable_surface_diagnostic_model_name = stable_best[
                (stable_best["cv_kind"] == "group_doi_or_reference")
                & (stable_best["feature_set"] == SURFACE_STABLE_DIAGNOSTIC_FEATURE_SET)
            ]["model"].iloc[0]
            stable_surface_diagnostic_features = STABLE_FEATURE_SETS[SURFACE_STABLE_DIAGNOSTIC_FEATURE_SET]
            stable_surface_diagnostic_model = make_model(stable_surface_diagnostic_model_name, stable_surface_diagnostic_features, df_stable)
            stable_surface_diagnostic_model.fit(df_stable[stable_surface_diagnostic_features].copy(), df_stable[STABLE_TARGET].copy())

            def train_app_target_bundle(target, label, data, features, purpose, deployed_feature_set):
                target_data = data.dropna(subset=[target]).copy()
                target_results = evaluate_feature_sets(
                    target_data,
                    target,
                    {deployed_feature_set: features},
                    "group_doi_or_reference",
                    CORE_MODELS,
                )
                target_best = best_by_feature_set(target_results)
                target_model_name = target_best[
                    (target_best["cv_kind"] == "group_doi_or_reference")
                    & (target_best["feature_set"] == deployed_feature_set)
                ]["model"].iloc[0]
                target_model = make_model(target_model_name, features, target_data)
                target_model.fit(target_data[features].copy(), target_data[target].copy())
                return {
                    "purpose": purpose,
                    "target": target,
                    "label": label,
                    "model": target_model,
                    "best_model_name": target_model_name,
                    "deployed_feature_set": deployed_feature_set,
                    "feature_columns": features,
                    "cv_results": target_results,
                    "best_by_set": target_best,
                    "n_training_rows": len(target_data),
                    "n_groups": target_data["doi_group"].nunique(),
                }

            discovery_bundle = {
                "purpose": "final leakage-controlled SiOC/SiOCN discovery model",
                "target": TARGET,
                "label": "First-cycle reversible capacity",
                "model": primary_model,
                "best_model_name": primary_model_name,
                "deployed_feature_set": DEPLOYED_FIRST_FEATURE_SET,
                "feature_columns": primary_features,
                "feature_sets": FIRST_FEATURE_SETS,
                "cv_results": first_results,
                "best_by_set": first_best,
                "oof_grouped": first_oof_grouped,
                "doi_group_audit": group_audit,
                "n_training_rows": len(df_first),
                "n_groups": df_first["doi_group"].nunique(),
            }
            first_surface_model_name = first_best[
                (first_best["cv_kind"] == "group_doi_or_reference")
                & (first_best["feature_set"] == SURFACE_FIRST_FEATURE_SET)
            ]["model"].iloc[0]
            first_surface_features = FIRST_FEATURE_SETS[SURFACE_FIRST_FEATURE_SET]
            first_surface_model = make_model(first_surface_model_name, first_surface_features, df_first)
            first_surface_model.fit(df_first[first_surface_features].copy(), df_first[TARGET].copy())
            first_surface_bundle = {
                "purpose": "optional surface-assisted SiOC/SiOCN first-cycle capacity model",
                "target": TARGET,
                "label": "First-cycle reversible capacity, surface-assisted",
                "model": first_surface_model,
                "best_model_name": first_surface_model_name,
                "deployed_feature_set": SURFACE_FIRST_FEATURE_SET,
                "feature_columns": first_surface_features,
                "feature_sets": FIRST_FEATURE_SETS,
                "cv_results": first_results,
                "best_by_set": first_best,
                "n_training_rows": len(df_first),
                "n_groups": df_first["doi_group"].nunique(),
            }
            stable_bundle = {
                "purpose": "stable/cycled SiOC/SiOCN capacity model",
                "target": STABLE_TARGET,
                "label": "Cycled reversible capacity at defined cycle number",
                "model": stable_primary_model,
                "best_model_name": stable_primary_model_name,
                "deployed_feature_set": DEPLOYED_STABLE_FEATURE_SET,
                "feature_columns": stable_primary_features,
                "feature_sets": STABLE_FEATURE_SETS,
                "cv_results": stable_results,
                "best_by_set": stable_best,
                "n_training_rows": len(df_stable),
                "n_groups": df_stable["doi_group"].nunique(),
            }
            stable_surface_bundle = {
                "purpose": "optional surface-assisted stable/cycled SiOC/SiOCN capacity model",
                "target": STABLE_TARGET,
                "label": "Cycled reversible capacity, surface-assisted",
                "model": stable_surface_model,
                "best_model_name": stable_surface_model_name,
                "deployed_feature_set": SURFACE_STABLE_FEATURE_SET,
                "feature_columns": stable_surface_features,
                "feature_sets": STABLE_FEATURE_SETS,
                "cv_results": stable_results,
                "best_by_set": stable_best,
                "n_training_rows": len(df_stable),
                "n_groups": df_stable["doi_group"].nunique(),
            }
            irreversible_bundle = train_app_target_bundle(
                "irreversible_capacity_mah_g",
                "First-cycle irreversible capacity",
                df_first,
                primary_features,
                "first-cycle irreversible-capacity model",
                DEPLOYED_FIRST_FEATURE_SET,
            )
            ce_bundle = train_app_target_bundle(
                "coulombic_efficiency_pct",
                "First-cycle Coulombic efficiency, design-stage",
                df_first,
                primary_features,
                "first-cycle Coulombic-efficiency model",
                DEPLOYED_FIRST_FEATURE_SET,
            )
            ce_diagnostic_features = primary_features + [TARGET]
            ce_diagnostic_bundle = train_app_target_bundle(
                "coulombic_efficiency_pct",
                "First-cycle Coulombic efficiency, diagnostic",
                df_first,
                ce_diagnostic_features,
                "first-cycle Coulombic-efficiency diagnostic model using first-cycle reversible capacity",
                "A_material_plus_first_reversible_capacity",
            )
            stable_diagnostic_bundle = {
                "purpose": "diagnostic stable/cycled SiOC/SiOCN capacity model using first-cycle reversible capacity",
                "target": STABLE_TARGET,
                "label": "Cycled reversible capacity, diagnostic",
                "model": stable_diagnostic_model,
                "best_model_name": stable_diagnostic_model_name,
                "deployed_feature_set": DEPLOYED_STABLE_DIAGNOSTIC_FEATURE_SET,
                "feature_columns": stable_diagnostic_features,
                "feature_sets": STABLE_FEATURE_SETS,
                "cv_results": stable_results,
                "best_by_set": stable_best,
                "n_training_rows": len(df_stable),
                "n_groups": df_stable["doi_group"].nunique(),
            }
            stable_surface_diagnostic_bundle = {
                "purpose": "optional surface-assisted diagnostic stable/cycled model using first-cycle reversible capacity",
                "target": STABLE_TARGET,
                "label": "Cycled reversible capacity, surface-assisted diagnostic",
                "model": stable_surface_diagnostic_model,
                "best_model_name": stable_surface_diagnostic_model_name,
                "deployed_feature_set": SURFACE_STABLE_DIAGNOSTIC_FEATURE_SET,
                "feature_columns": stable_surface_diagnostic_features,
                "feature_sets": STABLE_FEATURE_SETS,
                "cv_results": stable_results,
                "best_by_set": stable_best,
                "n_training_rows": len(df_stable),
                "n_groups": df_stable["doi_group"].nunique(),
            }
            app_bundles = {
                "first_reversible": discovery_bundle,
                "first_reversible_surface": first_surface_bundle,
                "ce_design": ce_bundle,
                "ce_diagnostic": ce_diagnostic_bundle,
                "stable_design": stable_bundle,
                "stable_design_surface": stable_surface_bundle,
                "stable_diagnostic": stable_diagnostic_bundle,
                "stable_diagnostic_surface": stable_surface_diagnostic_bundle,
                "irreversible_diagnostic_only": irreversible_bundle,
            }
            joblib.dump(discovery_bundle, MODEL_DIR / "sioc_final_discovery_model.joblib")
            joblib.dump(first_surface_bundle, MODEL_DIR / "sioc_final_discovery_surface_model.joblib")
            joblib.dump(stable_bundle, MODEL_DIR / "sioc_stable_capacity_model.joblib")
            joblib.dump(stable_surface_bundle, MODEL_DIR / "sioc_stable_capacity_surface_model.joblib")
            joblib.dump(stable_diagnostic_bundle, MODEL_DIR / "sioc_stable_capacity_diagnostic_model.joblib")
            joblib.dump(stable_surface_diagnostic_bundle, MODEL_DIR / "sioc_stable_capacity_surface_diagnostic_model.joblib")
            joblib.dump(irreversible_bundle, MODEL_DIR / "sioc_irreversible_capacity_model.joblib")
            joblib.dump(ce_bundle, MODEL_DIR / "sioc_coulombic_efficiency_model.joblib")
            joblib.dump(ce_diagnostic_bundle, MODEL_DIR / "sioc_coulombic_efficiency_diagnostic_model.joblib")
            joblib.dump(app_bundles, MODEL_DIR / "sioc_app_target_models.joblib")
            print("Saved discovery model:", MODEL_DIR / "sioc_final_discovery_model.joblib")
            print("Saved surface discovery model:", MODEL_DIR / "sioc_final_discovery_surface_model.joblib")
            print("Saved stable model:", MODEL_DIR / "sioc_stable_capacity_model.joblib")
            print("Saved surface stable model:", MODEL_DIR / "sioc_stable_capacity_surface_model.joblib")
            print("Saved stable diagnostic model:", MODEL_DIR / "sioc_stable_capacity_diagnostic_model.joblib")
            print("Saved surface stable diagnostic model:", MODEL_DIR / "sioc_stable_capacity_surface_diagnostic_model.joblib")
            print("Saved irreversible model:", MODEL_DIR / "sioc_irreversible_capacity_model.joblib")
            print("Saved CE model:", MODEL_DIR / "sioc_coulombic_efficiency_model.joblib")
            print("Saved CE diagnostic model:", MODEL_DIR / "sioc_coulombic_efficiency_diagnostic_model.joblib")
            print("Saved app target models:", MODEL_DIR / "sioc_app_target_models.joblib")
            print("Primary discovery model:", primary_model_name)
            print("Primary stable model:", stable_primary_model_name)
            print("Diagnostic stable model:", stable_diagnostic_bundle["best_model_name"])
            print("Primary irreversible model:", irreversible_bundle["best_model_name"])
            print("Primary CE model:", ce_bundle["best_model_name"])
            print("Diagnostic CE model:", ce_diagnostic_bundle["best_model_name"])
            """
        ),
        md(
            """
            ## 14. Final Recommendation

            Recommended model for GitHub/project reporting:

            - **Primary app/deployed model:** `B_composition_T_time`, trained on first reported reversible capacity. It uses only elemental composition (Si/C/O/N wt.%) and final pyrolysis temperature/time.
            - **Interpretation of capacity:** predictions should be read as low-current literature-regime capacities, approximately **0.05-0.1C graphite-equivalent** and measured in a **0-3 V voltage window**. Most rows are near 18-18.6 mA/g, i.e. about 0.05C when normalized to graphite's 372 mAh/g capacity, and most use 0-3 V.
            - **Atmosphere diagnostic:** `B_material_plus_atmosphere_group` groups primary pyrolysis atmosphere as inert/protective (Ar/N2/Ar-H2/unknown/reported inert, including Ar pyrolysis followed by air treatment), reducing (pure H2), and oxidizing (CO2/O2 if primary pyrolysis). It is retained for discussion, but not deployed because 212/219 first-capacity rows (~97%) are interpreted as primary inert/protective-atmosphere pyrolysis and the non-inert classes are small.
            - **Optional surface-area model:** `D_surface_assisted_composition_T_time`, retained when measured BET surface area is available. It uses raw `surface_area_m2_g`, not log(surface area).
            - **Polymer/precursor role:** polymer family is excluded from the predictive model and kept only for chemistry-aware route recommendation.
            - **Similarity/routing distance:** literature analog and route-family matching use only elemental composition distance (Si/C/O/N wt.%), so recipes are described as composition-matched guides rather than copied performance surrogates.
            - **Excluded from deployed ML features:** polymer family, DVB loading, pre-pyrolysis/crosslinking parameters, voltage window, current density, and pyrolysis atmosphere. They can still be discussed as literature/process/protocol context, but they are not used for prediction.
	            - **Primary validation:** DOI/reference-grouped CV, because it evaluates transfer to unseen DOI/reference groups.
	            - **Secondary validation:** shuffled CV, reported only as an optimistic interpolation diagnostic. It is not used for deployed model selection because rows from the same DOI/source can be split between train and test.
            - **Stable/cycled model:** keep as a secondary analysis because the stable target has fewer rows and stronger protocol heterogeneity.
            - **Diagnostic stable model:** first capacity helps predict cycled capacity, but it is not a pure pre-test discovery model.
            - **First-cycle CE and irreversible capacity:** deploy CE as a model target and calculate irreversible capacity from `Qirrev = Qrev * (100 / CE% - 1)`. A diagnostic CE model using `reversible_capacity_mah_g` is retained for post-first-cycle use.
            - **Apparent retention:** calculate `100 * Qcycled / Qrev` rather than deploying a standalone retention model. Report it as protocol-dependent apparent retention because first-cycle and cycled capacities may be measured at different current densities.
            - **ANN/MLP:** tested and retained as a benchmark, but not selected because it underperforms tree ensembles in this small-data regime.
            - **PCA/SHAP/correlations:** used for interpretation and data-quality discussion, not as replacements for physically meaningful descriptors.
            """
        ),
    ]

    nb["cells"] = cells
    out = Path("notebooks/05_final_clean_sioc_discovery_modeling.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
