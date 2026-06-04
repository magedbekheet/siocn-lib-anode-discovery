"""Train final Streamlit app models for the simplified SiOC descriptor policy."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import TARGET, prepare_features, read_sioc_csv  # noqa: E402
from train_discovery_model import make_groups  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "sioc_battery_capacity_clean_updated.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
STABLE_TARGET = "cycling_reversible_capacity_mah_g"
CE_TARGET = "coulombic_efficiency_pct"
IRREVERSIBLE_TARGET = "irreversible_capacity_mah_g"

COMPOSITION_FEATURES = ["si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct"]
TEMP = ["pyrolysis_temp_c"]
TIME = ["pyrolysis_time_h"]
SURFACE = ["surface_area_m2_g"]
CYCLE = ["cycling_numbers"]

FIRST_FEATURE_SETS = {
    "A_composition_T": COMPOSITION_FEATURES + TEMP,
    "B_composition_T_time": COMPOSITION_FEATURES + TEMP + TIME,
    "C_surface_assisted_composition_T": COMPOSITION_FEATURES + TEMP + SURFACE,
    "D_surface_assisted_composition_T_time": COMPOSITION_FEATURES + TEMP + TIME + SURFACE,
}
STABLE_FEATURE_SETS = {
    "A_stable_composition_T_cycle": COMPOSITION_FEATURES + TEMP + CYCLE,
    "B_stable_composition_T_time_cycle": COMPOSITION_FEATURES + TEMP + TIME + CYCLE,
    "C_surface_assisted_stable_composition_T_cycle": COMPOSITION_FEATURES + TEMP + CYCLE + SURFACE,
    "D_surface_assisted_stable_composition_T_time_cycle": COMPOSITION_FEATURES + TEMP + TIME + CYCLE + SURFACE,
    "E_stable_plus_first_capacity_diagnostic": COMPOSITION_FEATURES + TEMP + TIME + CYCLE + [TARGET],
    "F_surface_assisted_stable_plus_first_capacity_diagnostic": COMPOSITION_FEATURES + TEMP + TIME + CYCLE + SURFACE + [TARGET],
}

DEPLOYED_FIRST = "B_composition_T_time"
SURFACE_FIRST = "D_surface_assisted_composition_T_time"
DEPLOYED_STABLE = "B_stable_composition_T_time_cycle"
SURFACE_STABLE = "D_surface_assisted_stable_composition_T_time_cycle"
DEPLOYED_STABLE_DIAG = "E_stable_plus_first_capacity_diagnostic"
SURFACE_STABLE_DIAG = "F_surface_assisted_stable_plus_first_capacity_diagnostic"


def public_polymer_template(family: str, dvb_modification: int = 0) -> str:
    family_text = str(family).replace("_", " ").lower()
    templates = [
        ("polysilazane", "polyorganosilazane / polysilazane"),
        ("silylcarbodiimide", "polyphenylvinylsilylcarbodiimide"),
        ("polycarbosilane", "polycarbosilane PCS"),
        ("polysilsesquioxane", "polysilsesquioxane PMS-derived resin"),
        ("phtes teos", "phenyltriethoxysilane PhTES + tetraethoxysilane TEOS"),
        ("prtes teos", "triethoxypropylsilane PrTES + tetraethoxysilane TEOS"),
        ("phtes mtes", "phenyltriethoxysilane PhTES + methyltriethoxysilane MTES"),
        ("phtes vtes", "phenyltriethoxysilane PhTES + vinyltriethoxysilane VTES"),
        ("vtes mtes", "vinyltriethoxysilane VTES + methyltriethoxysilane MTES"),
        ("phtes", "phenyltriethoxysilane PhTES-derived polysiloxane"),
        ("vtes", "vinyltriethoxysilane VTES-derived polysiloxane"),
        ("mtes", "methyltriethoxysilane MTES-derived polysiloxane"),
        ("teos", "tetraethoxysilane TEOS-derived silica/polysiloxane"),
        ("phenylalkyl polysiloxane", "methylvinylphenyl polysiloxane resin"),
        ("phenyl polysiloxane", "phenyl polysiloxane resin"),
        ("vinyl polysiloxane", "vinyl polysiloxane resin"),
        ("alkyl polysiloxane", "alkyl/methyl polysiloxane resin"),
        ("silicone oil", "silicone oil"),
        ("carbon rich", "carbon-rich precursor blend"),
        ("organopolysilane", "organopolysilane copolymer"),
        ("polysiloxane", "generic polysiloxane resin"),
    ]
    base = "generic polymer-derived ceramic precursor"
    for token, text in templates:
        if token in family_text:
            base = text
            break
    if int(dvb_modification or 0):
        return f"{base} + DVB"
    return base


def build_public_reference_library(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate route/prediction context without exposing row-level literature data."""
    group_cols = ["polymer_family_broad", "dvb_modification"]
    required = ["polymer_family_broad", "dvb_modification", *COMPOSITION_FEATURES, "pyrolysis_temp_c", "pyrolysis_time_h"]
    usable = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    if usable.empty:
        return pd.DataFrame()

    numeric_cols = [
        *COMPOSITION_FEATURES,
        "pyrolysis_temp_c",
        "pyrolysis_time_h",
        "surface_area_m2_g",
        "cycling_numbers",
        "first_cycling_current_ma_g",
        "cycling_current_ma_g",
        TARGET,
        STABLE_TARGET,
        CE_TARGET,
        IRREVERSIBLE_TARGET,
    ]
    rows = []
    for keys, group in usable.groupby(group_cols, dropna=False):
        family, dvb_mod = keys
        dvb_mod = int(0 if pd.isna(dvb_mod) else dvb_mod)
        row = {
            "polymer": public_polymer_template(str(family), dvb_mod),
            "polymer_family_broad": str(family),
            "dvb_modification": dvb_mod,
            "pyrolysis_atmosphere": "inert",
            "pre_pyrolysis_method": "none",
            "crosslinking_method": "none",
            "crosslink_temp_c": np.nan,
            "crosslink_time_h": np.nan,
            "crosslink_atmosphere": "unknown",
            "voltage_min_v": 0.0,
            "voltage_max_v": 3.0,
            "voltage_window_v": 3.0,
            "reference": f"Public aggregate route family, n={len(group)}",
            "doi": "",
            "sample_count": int(len(group)),
            "is_public_aggregate": True,
        }
        for col in numeric_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                row[col] = float(values.median()) if not values.empty else np.nan
        rows.append(row)

    public_ref = pd.DataFrame(rows)
    public_ref = public_ref.sort_values(["sample_count", "polymer_family_broad"], ascending=[False, True]).reset_index(drop=True)
    return public_ref


def make_model(name: str, cols: list[str]) -> Pipeline:
    models = {
        "Baseline mean": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=6,
            subsample=0.85,
            random_state=RANDOM_STATE,
        ),
    }
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                cols,
            )
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", pre), ("regressor", models[name])])


def cv_scores(data: pd.DataFrame, target: str, cols: list[str], model_name: str, cv_kind: str) -> dict:
    X = data[cols].copy()
    y = data[target].copy()
    if cv_kind == "grouped":
        groups = data["doi_group"]
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        splits = cv.split(X, y, groups)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = cv.split(X, y)

    preds = np.full(len(data), np.nan)
    for train_idx, test_idx in splits:
        model = make_model(model_name, cols)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = model.predict(X.iloc[test_idx])
    return {
        "cv_kind": cv_kind,
        "model": model_name,
        "mae_mean": mean_absolute_error(y, preds),
        "test_r2_mean": r2_score(y, preds),
    }


def evaluate(data: pd.DataFrame, target: str, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature_set, cols in feature_sets.items():
        cols = [c for c in cols if c in data.columns]
        for model_name in ["Baseline mean", "Ridge", "Random Forest", "Extra Trees", "Gradient Boosting"]:
            for cv_kind in ["grouped", "shuffled"]:
                scores = cv_scores(data, target, cols, model_name, cv_kind)
                rows.append(
                    {
                        "target": target,
                        "feature_set": feature_set,
                        "n_rows": len(data),
                        "n_groups": data["doi_group"].nunique(),
                        "n_features_raw": len(cols),
                        **scores,
                    }
                )
    results = pd.DataFrame(rows)
    best = (
        results[results["model"] != "Baseline mean"]
        .sort_values(["cv_kind", "feature_set", "mae_mean"])
        .groupby(["cv_kind", "feature_set"], as_index=False)
        .first()
        .sort_values(["cv_kind", "mae_mean"])
    )
    return results, best


def best_model_name(best: pd.DataFrame, feature_set: str) -> str:
    row = best[(best["cv_kind"] == "grouped") & (best["feature_set"] == feature_set)].sort_values("mae_mean").iloc[0]
    return str(row["model"])


def train_bundle(
    data: pd.DataFrame,
    target: str,
    label: str,
    purpose: str,
    feature_set: str,
    feature_sets: dict[str, list[str]],
    results: pd.DataFrame,
    best: pd.DataFrame,
    forced_model_name: str | None = None,
) -> dict:
    cols = [c for c in feature_sets[feature_set] if c in data.columns]
    model_name = forced_model_name or best_model_name(best, feature_set)
    model = make_model(model_name, cols)
    model.fit(data[cols].copy(), data[target].copy())
    return {
        "purpose": purpose,
        "target": target,
        "label": label,
        "model": model,
        "best_model_name": model_name,
        "deployed_feature_set": feature_set,
        "feature_columns": cols,
        "feature_sets": feature_sets,
        "cv_results": results,
        "best_by_set": best,
        "n_training_rows": len(data),
        "n_groups": data["doi_group"].nunique(),
    }


def main() -> None:
    raw = read_sioc_csv(str(DATA_PATH))
    df = prepare_features(raw)
    df_first = df.dropna(subset=[TARGET]).copy()
    df_first["doi_group"] = make_groups(df_first).values
    df_stable = df.dropna(subset=[STABLE_TARGET, "cycling_numbers"]).copy()
    df_stable["doi_group"] = make_groups(df_stable).values

    first_results, first_best = evaluate(df_first, TARGET, FIRST_FEATURE_SETS)
    stable_results, stable_best = evaluate(df_stable, STABLE_TARGET, STABLE_FEATURE_SETS)

    ce_data = df_first.dropna(subset=[CE_TARGET]).copy()
    ce_data["doi_group"] = make_groups(ce_data).values
    ce_feature_sets = {
        DEPLOYED_FIRST: FIRST_FEATURE_SETS[DEPLOYED_FIRST],
        "A_material_plus_first_reversible_capacity": FIRST_FEATURE_SETS[DEPLOYED_FIRST] + [TARGET],
    }
    ce_results, ce_best = evaluate(ce_data, CE_TARGET, ce_feature_sets)

    irrev_data = df_first.dropna(subset=[IRREVERSIBLE_TARGET]).copy()
    irrev_data["doi_group"] = make_groups(irrev_data).values
    irrev_results, irrev_best = evaluate(irrev_data, IRREVERSIBLE_TARGET, {DEPLOYED_FIRST: FIRST_FEATURE_SETS[DEPLOYED_FIRST]})

    app_bundles = {
        "first_reversible": train_bundle(
            df_first, TARGET, "First-cycle reversible capacity",
            "final simplified SiOC/SiOCN first-cycle capacity model",
            DEPLOYED_FIRST, FIRST_FEATURE_SETS, first_results, first_best,
        ),
        "first_reversible_surface": train_bundle(
            df_first, TARGET, "First-cycle reversible capacity, surface-assisted",
            "optional raw-surface-assisted first-cycle capacity model",
            SURFACE_FIRST, FIRST_FEATURE_SETS, first_results, first_best,
        ),
        "ce_design": train_bundle(
            ce_data, CE_TARGET, "First-cycle Coulombic efficiency, design-stage",
            "first-cycle CE model",
            DEPLOYED_FIRST, ce_feature_sets, ce_results, ce_best,
        ),
        "ce_diagnostic": train_bundle(
            ce_data, CE_TARGET, "First-cycle Coulombic efficiency, diagnostic",
            "first-cycle CE diagnostic model using Qrev",
            "A_material_plus_first_reversible_capacity", ce_feature_sets, ce_results, ce_best,
        ),
        "stable_design": train_bundle(
            df_stable, STABLE_TARGET, "Cycled reversible capacity",
            "stable/cycled capacity model",
            DEPLOYED_STABLE, STABLE_FEATURE_SETS, stable_results, stable_best,
        ),
        "stable_design_surface": train_bundle(
            df_stable, STABLE_TARGET, "Cycled reversible capacity, surface-assisted",
            "optional raw-surface-assisted cycled capacity model",
            SURFACE_STABLE, STABLE_FEATURE_SETS, stable_results, stable_best,
        ),
        "stable_diagnostic": train_bundle(
            df_stable, STABLE_TARGET, "Cycled reversible capacity, diagnostic",
            "diagnostic stable/cycled capacity model using Qrev",
            DEPLOYED_STABLE_DIAG, STABLE_FEATURE_SETS, stable_results, stable_best,
        ),
        "stable_diagnostic_surface": train_bundle(
            df_stable, STABLE_TARGET, "Cycled reversible capacity, surface-assisted diagnostic",
            "optional raw-surface-assisted diagnostic cycled capacity model using Qrev",
            SURFACE_STABLE_DIAG, STABLE_FEATURE_SETS, stable_results, stable_best,
        ),
        "irreversible_diagnostic_only": train_bundle(
            irrev_data, IRREVERSIBLE_TARGET, "First-cycle irreversible capacity",
            "diagnostic irreversible-capacity model retained for comparison",
            DEPLOYED_FIRST, {DEPLOYED_FIRST: FIRST_FEATURE_SETS[DEPLOYED_FIRST]}, irrev_results, irrev_best,
        ),
    }
    app_bundles["_public_reference_raw"] = build_public_reference_library(df)

    joblib.dump(app_bundles["first_reversible"], MODEL_DIR / "sioc_final_discovery_model.joblib")
    joblib.dump(app_bundles["first_reversible_surface"], MODEL_DIR / "sioc_final_discovery_surface_model.joblib")
    joblib.dump(app_bundles["stable_design"], MODEL_DIR / "sioc_stable_capacity_model.joblib")
    joblib.dump(app_bundles["stable_design_surface"], MODEL_DIR / "sioc_stable_capacity_surface_model.joblib")
    joblib.dump(app_bundles["stable_diagnostic"], MODEL_DIR / "sioc_stable_capacity_diagnostic_model.joblib")
    joblib.dump(app_bundles["stable_diagnostic_surface"], MODEL_DIR / "sioc_stable_capacity_surface_diagnostic_model.joblib")
    joblib.dump(app_bundles["ce_design"], MODEL_DIR / "sioc_coulombic_efficiency_model.joblib")
    joblib.dump(app_bundles["ce_diagnostic"], MODEL_DIR / "sioc_coulombic_efficiency_diagnostic_model.joblib")
    joblib.dump(app_bundles["irreversible_diagnostic_only"], MODEL_DIR / "sioc_irreversible_capacity_model.joblib")
    joblib.dump(app_bundles, MODEL_DIR / "sioc_app_target_models.joblib")

    summary = pd.concat([first_results, stable_results, ce_results, irrev_results], ignore_index=True)
    best_summary = pd.concat([first_best, stable_best, ce_best, irrev_best], ignore_index=True)
    summary.to_csv(REPORT_DIR / "final_simplified_model_cv_results.csv", index=False)
    best_summary.to_csv(REPORT_DIR / "final_simplified_model_best_by_feature_set.csv", index=False)
    app_bundles["_public_reference_raw"].to_csv(REPORT_DIR / "public_aggregate_reference_library.csv", index=False)

    print("Saved app bundle:", MODEL_DIR / "sioc_app_target_models.joblib")
    print("Public aggregate reference rows:", len(app_bundles["_public_reference_raw"]))
    print(best_summary[["target", "cv_kind", "feature_set", "model", "mae_mean", "test_r2_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
