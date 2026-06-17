"""Self-driving-lab candidate generation for SiOC/SiOCN anode discovery.

This module is a decision-support prototype. It ranks candidate compositions
and pyrolysis conditions, exports structured experiment manifests, and keeps
precursor selection and hardware execution under explicit human review.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import erf, pi, sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from src.features import TARGET, prepare_features


PUBLIC_ANALOG_KEY = "_public_literature_analogs_raw"
PUBLIC_RANGE_STATS_KEY = "_public_range_stats"
CE_TARGET = "coulombic_efficiency_pct"
STABLE_TARGET = "cycling_reversible_capacity_mah_g"

COMPOSITION_COLUMNS = ["si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct"]
DESIGN_COLUMNS = COMPOSITION_COLUMNS + ["pyrolysis_temp_c", "pyrolysis_time_h"]
CHARACTERIZATION_FIELDS = [
    "crystallinity_pct",
    "grain_size_nm",
    "bet_surface_area_m2_g",
    "raman_d_g_ratio",
    "pore_volume_cm3_g",
]
MEASUREMENT_METADATA_FIELDS = [
    "replicate_id",
    "measurement_status",
    "measurement_quality_flag",
]
MEASURED_TARGET_FIELDS = [
    "measured_qrev_mah_g",
    "measured_ce_pct",
    "measured_qcycled_mah_g",
]
ALLOWED_MEASUREMENT_STATUSES = {"completed", "failed", "partial", "excluded"}
ALLOWED_MEASUREMENT_QUALITY_FLAGS = {"pass", "warning", "fail", "not_assessed"}
NOVELTY_METHODS = {"legacy", "mahalanobis", "kde", "hybrid"}

DEFAULT_MEASUREMENT_PLAN = [
    "elemental analysis (Si/C/O/N)",
    "Raman spectroscopy",
    "X-ray diffraction",
    "BET surface area / gas sorption",
    "SEM or TEM imaging",
    "first-cycle reversible capacity and Coulombic efficiency",
    "cycled capacity at the selected cycle number",
]


def load_sdl_bundle(path: str | Path) -> dict[str, Any]:
    """Load the public bundle and enforce the pre-synthesis model boundary."""
    bundle = joblib.load(Path(path))
    required = ["first_reversible", "ce_design", "stable_design", PUBLIC_ANALOG_KEY]
    missing = [key for key in required if key not in bundle]
    if missing:
        raise KeyError(f"Model bundle is missing required SDL objects: {missing}")
    analogs = bundle[PUBLIC_ANALOG_KEY]
    if not isinstance(analogs, pd.DataFrame) or analogs.empty:
        raise ValueError("The public literature analog table is missing or empty.")
    assert_characterization_not_in_production_features(bundle)
    return bundle


def assert_characterization_not_in_production_features(bundle: dict[str, Any]) -> None:
    """Confirm measured characterization fields cannot enter deployed predictors.

    The closed-loop layer may store these fields for provenance and future model
    development. Current production capacity and efficiency models remain
    pre-synthesis predictors and must not consume post-synthesis measurements.
    """
    violations: dict[str, list[str]] = {}
    for model_name in ("first_reversible", "ce_design", "stable_design"):
        feature_columns = set(bundle.get(model_name, {}).get("feature_columns", []))
        overlap = sorted(feature_columns.intersection(CHARACTERIZATION_FIELDS))
        if overlap:
            violations[model_name] = overlap
    if violations:
        raise ValueError(
            "Characterization fields are not allowed in production prediction features: "
            f"{violations}"
        )


def validate_measurement_table(measurements: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize closed-loop measurements without model leakage.

    A measurement row must identify a proposed experiment through
    ``experiment_id`` or ``candidate_id``. Completed rows require measured Qrev
    because only Qrev updates the GP acquisition/reference layer. Optional
    characterization descriptors are retained for storage and provenance only.
    """
    if not isinstance(measurements, pd.DataFrame) or measurements.empty:
        raise ValueError("Measurement input must be a non-empty table.")

    result = measurements.copy()
    if "experiment_id" not in result.columns and "candidate_id" not in result.columns:
        raise ValueError("Measurements require experiment_id or candidate_id.")
    if "experiment_id" not in result.columns:
        result["experiment_id"] = result["candidate_id"]
    elif "candidate_id" in result.columns:
        result["experiment_id"] = result["experiment_id"].fillna(result["candidate_id"])

    result["experiment_id"] = result["experiment_id"].fillna("").astype(str).str.strip()
    if (result["experiment_id"] == "").any():
        raise ValueError("Measurement identifiers cannot be empty.")

    if "replicate_id" not in result.columns:
        result["replicate_id"] = "replicate-1"
    result["replicate_id"] = result["replicate_id"].fillna("replicate-1").astype(str).str.strip()
    if (result["replicate_id"] == "").any():
        raise ValueError("replicate_id cannot be empty.")

    if "measurement_status" not in result.columns:
        result["measurement_status"] = "completed"
    result["measurement_status"] = (
        result["measurement_status"].fillna("completed").astype(str).str.strip().str.lower()
    )
    invalid_statuses = sorted(set(result["measurement_status"]) - ALLOWED_MEASUREMENT_STATUSES)
    if invalid_statuses:
        raise ValueError(
            f"Unsupported measurement_status values {invalid_statuses}; "
            f"allowed values are {sorted(ALLOWED_MEASUREMENT_STATUSES)}."
        )

    if "measurement_quality_flag" not in result.columns:
        result["measurement_quality_flag"] = "not_assessed"
    result["measurement_quality_flag"] = (
        result["measurement_quality_flag"]
        .fillna("not_assessed")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    invalid_flags = sorted(
        set(result["measurement_quality_flag"]) - ALLOWED_MEASUREMENT_QUALITY_FLAGS
    )
    if invalid_flags:
        raise ValueError(
            f"Unsupported measurement_quality_flag values {invalid_flags}; "
            f"allowed values are {sorted(ALLOWED_MEASUREMENT_QUALITY_FLAGS)}."
        )

    if "measured_qrev_mah_g" not in result.columns:
        result["measured_qrev_mah_g"] = np.nan
    for column in [*MEASURED_TARGET_FIELDS, *CHARACTERIZATION_FIELDS]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    completed_without_qrev = result["measurement_status"].eq("completed") & result[
        "measured_qrev_mah_g"
    ].isna()
    if completed_without_qrev.any():
        bad_ids = result.loc[completed_without_qrev, "experiment_id"].tolist()
        raise ValueError(f"Completed measurements require measured_qrev_mah_g: {bad_ids}")

    if "crystallinity_pct" in result.columns:
        invalid = result["crystallinity_pct"].dropna()
        if ((invalid < 0) | (invalid > 100)).any():
            raise ValueError("crystallinity_pct must be between 0 and 100.")
    for column in [
        "grain_size_nm",
        "bet_surface_area_m2_g",
        "raman_d_g_ratio",
        "pore_volume_cm3_g",
        *MEASURED_TARGET_FIELDS,
    ]:
        if column in result.columns and (result[column].dropna() < 0).any():
            raise ValueError(f"{column} cannot be negative.")

    duplicate_mask = result.duplicated(["experiment_id", "replicate_id"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = result.loc[
            duplicate_mask, ["experiment_id", "replicate_id"]
        ].to_dict("records")
        raise ValueError(f"Duplicate experiment/replicate rows found: {duplicate_keys}")
    return result


def _numeric_reference(
    bundle: dict[str, Any],
    reference_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source = bundle[PUBLIC_ANALOG_KEY] if reference_override is None else reference_override
    if not isinstance(source, pd.DataFrame) or source.empty:
        raise ValueError("The acquisition reference table is missing or empty.")
    reference = source.copy().reset_index(drop=True)
    numeric_cols = [
        *DESIGN_COLUMNS,
        TARGET,
        CE_TARGET,
        STABLE_TARGET,
        "irreversible_capacity_mah_g",
        "cycling_numbers",
    ]
    for col in numeric_cols:
        if col in reference.columns:
            reference[col] = pd.to_numeric(reference[col], errors="coerce")
    return reference


def join_measurements_to_plan(
    measurements: pd.DataFrame,
    planned_experiments: pd.DataFrame,
) -> pd.DataFrame:
    """Join measured outcomes to immutable planned design variables.

    Planned Si/C/O/N composition and pyrolysis conditions are authoritative.
    Characterization values remain attached to the measurement rows but are
    not included in the production-prediction or GP feature matrices.
    """
    validated = validate_measurement_table(measurements)
    if not isinstance(planned_experiments, pd.DataFrame) or planned_experiments.empty:
        raise ValueError("The prior recommendation table is missing or empty.")

    plan = planned_experiments.copy()
    if "experiment_id" not in plan.columns and "candidate_id" not in plan.columns:
        raise ValueError("Prior recommendations require experiment_id or candidate_id.")
    if "experiment_id" not in plan.columns:
        plan["experiment_id"] = plan["candidate_id"]
    plan["experiment_id"] = plan["experiment_id"].fillna("").astype(str).str.strip()
    missing_design = [column for column in DESIGN_COLUMNS if column not in plan.columns]
    if missing_design:
        raise ValueError(f"Prior recommendations are missing design columns: {missing_design}")
    if plan["experiment_id"].duplicated().any():
        raise ValueError("Prior recommendation identifiers must be unique.")

    plan[DESIGN_COLUMNS] = plan[DESIGN_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if plan[DESIGN_COLUMNS].isna().any().any():
        raise ValueError("Prior recommendation design variables must be numeric and complete.")
    composition_sums = plan[COMPOSITION_COLUMNS].sum(axis=1)
    if not np.allclose(composition_sums, 100.0, atol=0.15):
        bad_ids = plan.loc[~np.isclose(composition_sums, 100.0, atol=0.15), "experiment_id"]
        raise ValueError(f"Planned elemental compositions must sum to 100 wt.%: {bad_ids.tolist()}")

    unknown_ids = sorted(set(validated["experiment_id"]) - set(plan["experiment_id"]))
    if unknown_ids:
        raise ValueError(f"Measurements reference unknown experiments: {unknown_ids}")

    validated = validated.drop(columns=[column for column in DESIGN_COLUMNS if column in validated])
    plan_columns = ["experiment_id", *DESIGN_COLUMNS]
    for optional in ("candidate_origin", "recommendation_rank"):
        if optional in plan.columns:
            plan_columns.append(optional)
    return validated.merge(plan[plan_columns], on="experiment_id", how="left", validate="many_to_one")


def summarize_measurement_replicates(joined_measurements: pd.DataFrame) -> pd.DataFrame:
    """Summarize replicate outcomes while retaining failure counts."""
    if joined_measurements.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for experiment_id, group in joined_measurements.groupby("experiment_id", sort=True):
        record: dict[str, Any] = {
            "experiment_id": str(experiment_id),
            "replicate_count": int(len(group)),
            "completed_replicates": int(group["measurement_status"].eq("completed").sum()),
            "failed_replicates": int(group["measurement_status"].eq("failed").sum()),
        }
        for column in [*MEASURED_TARGET_FIELDS, *CHARACTERIZATION_FIELDS]:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            record[f"{column}_mean"] = float(values.mean())
            record[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records)


def build_updated_acquisition_reference(
    bundle: dict[str, Any],
    joined_measurements: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Append accepted measured Qrev means to the GP/reference layer only.

    This function does not fit, alter, or replace any deployed capacity model.
    The fixed production predictors remain unchanged. Completed measurements
    with non-failing quality flags update only the lightweight GP acquisition
    model and the novelty/domain reference used to rank the next candidates.
    """
    base = _numeric_reference(bundle)
    accepted = joined_measurements[
        joined_measurements["measurement_status"].eq("completed")
        & ~joined_measurements["measurement_quality_flag"].eq("fail")
        & joined_measurements["measured_qrev_mah_g"].notna()
    ].copy()

    replicate_stats = summarize_measurement_replicates(joined_measurements)
    measurement_reference = pd.DataFrame()
    if not accepted.empty:
        aggregation: dict[str, str] = {
            **{column: "first" for column in DESIGN_COLUMNS},
            "measured_qrev_mah_g": "mean",
        }
        for column in ["measured_ce_pct", "measured_qcycled_mah_g", *CHARACTERIZATION_FIELDS]:
            if column in accepted.columns:
                aggregation[column] = "mean"
        measurement_reference = accepted.groupby("experiment_id", as_index=False).agg(aggregation)
        measurement_reference[TARGET] = measurement_reference.pop("measured_qrev_mah_g")
        if "measured_ce_pct" in measurement_reference:
            measurement_reference[CE_TARGET] = measurement_reference.pop("measured_ce_pct")
        if "measured_qcycled_mah_g" in measurement_reference:
            measurement_reference[STABLE_TARGET] = measurement_reference.pop(
                "measured_qcycled_mah_g"
            )
        measurement_reference["reference_source"] = "closed_loop_measurement"

    combined = pd.concat([base, measurement_reference], ignore_index=True, sort=False)
    counts = {
        "public_reference_rows": int(len(base.dropna(subset=[*DESIGN_COLUMNS, TARGET]))),
        "accepted_measurement_replicates": int(len(accepted)),
        "accepted_measurement_experiments": int(len(measurement_reference)),
        "combined_reference_rows": int(len(combined.dropna(subset=[*DESIGN_COLUMNS, TARGET]))),
    }
    return combined, counts, replicate_stats


def _range_value(
    bundle: dict[str, Any],
    reference: pd.DataFrame,
    column: str,
    statistic: str,
    fallback: float,
) -> float:
    stats = bundle.get(PUBLIC_RANGE_STATS_KEY, {})
    if column in stats and statistic in stats[column]:
        value = float(stats[column][statistic])
        if np.isfinite(value):
            return value
    values = pd.to_numeric(reference.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return fallback
    lookup = {
        "min": values.min,
        "q05": lambda: values.quantile(0.05),
        "median": values.median,
        "q95": lambda: values.quantile(0.95),
        "max": values.max,
    }
    return float(lookup[statistic]())


def _repair_compositions(
    values: np.ndarray,
    min_si_wt_pct: float,
    min_c_wt_pct: float,
    max_n_wt_pct: float,
) -> np.ndarray:
    """Project compositions onto a simple SiOC/SiOCN feasibility region."""
    repaired = np.empty_like(values, dtype=float)
    for index, row in enumerate(np.clip(values, 0.0, None)):
        n_value = min(float(row[3]), max_n_wt_pct)
        remaining = 100.0 - n_value - min_si_wt_pct - min_c_wt_pct
        if remaining <= 0:
            raise ValueError("Composition bounds leave no room for Si/C/O allocation.")
        excess = np.array(
            [
                max(float(row[0]) - min_si_wt_pct, 0.0),
                max(float(row[1]) - min_c_wt_pct, 0.0),
                max(float(row[2]), 0.0),
            ]
        )
        if excess.sum() <= 0:
            excess = np.array([1.0, 1.0, 1.0])
        allocation = remaining * excess / excess.sum()
        repaired[index] = np.array(
            [
                min_si_wt_pct + allocation[0],
                min_c_wt_pct + allocation[1],
                allocation[2],
                n_value,
            ]
        )
    return repaired


def generate_candidate_space(
    bundle: dict[str, Any],
    n_candidates: int = 2500,
    exploration_fraction: float = 0.40,
    cycle_number: int = 100,
    min_si_wt_pct: float = 5.0,
    min_c_wt_pct: float = 5.0,
    max_n_wt_pct: float = 26.0,
    min_pyrolysis_temp_c: float = 800.0,
    max_pyrolysis_temp_c: float = 1400.0,
    min_pyrolysis_time_h: float = 0.5,
    max_pyrolysis_time_h: float = 6.0,
    random_state: int = 42,
    reference_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate local and exploratory candidates with compositions summing to 100 wt.%."""
    if n_candidates < 10:
        raise ValueError("n_candidates must be at least 10.")
    if not 0.0 <= exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be between 0 and 1.")
    if min_pyrolysis_temp_c >= max_pyrolysis_temp_c:
        raise ValueError("Minimum pyrolysis temperature must be below the maximum.")
    if min_pyrolysis_time_h >= max_pyrolysis_time_h:
        raise ValueError("Minimum pyrolysis time must be below the maximum.")

    reference = _numeric_reference(bundle, reference_override=reference_override)
    usable = reference.dropna(subset=DESIGN_COLUMNS).copy()
    if usable.empty:
        raise ValueError("No complete public reference rows are available for candidate generation.")

    rng = np.random.default_rng(random_state)
    n_explore = int(round(n_candidates * exploration_fraction))
    n_local = n_candidates - n_explore

    local_idx = rng.integers(0, len(usable), size=n_local)
    local_base = usable.iloc[local_idx]
    comp_std = usable[COMPOSITION_COLUMNS].std(ddof=0).replace(0, 1.0).to_numpy()
    local_comp = local_base[COMPOSITION_COLUMNS].to_numpy(float)
    local_comp += rng.normal(0.0, 0.16 * comp_std, size=local_comp.shape)
    local_comp = _repair_compositions(
        local_comp,
        min_si_wt_pct=min_si_wt_pct,
        min_c_wt_pct=min_c_wt_pct,
        max_n_wt_pct=max_n_wt_pct,
    )

    explore_n = rng.uniform(0.0, max_n_wt_pct, size=n_explore)
    zero_n_mask = rng.random(n_explore) < 0.25
    explore_n[zero_n_mask] = 0.0
    remaining = 100.0 - explore_n
    explore_sico = rng.dirichlet(np.array([2.8, 3.5, 1.8]), size=n_explore) * remaining[:, None]
    explore_comp = np.column_stack([explore_sico, explore_n])
    explore_comp = _repair_compositions(
        explore_comp,
        min_si_wt_pct=min_si_wt_pct,
        min_c_wt_pct=min_c_wt_pct,
        max_n_wt_pct=max_n_wt_pct,
    )

    temp_q05 = _range_value(bundle, reference, "pyrolysis_temp_c", "q05", 800.0)
    temp_q95 = _range_value(bundle, reference, "pyrolysis_temp_c", "q95", 1400.0)
    time_q05 = _range_value(bundle, reference, "pyrolysis_time_h", "q05", 0.5)
    time_q95 = _range_value(bundle, reference, "pyrolysis_time_h", "q95", 5.0)

    local_temp = local_base["pyrolysis_temp_c"].to_numpy(float) + rng.normal(0.0, 70.0, n_local)
    local_time = local_base["pyrolysis_time_h"].to_numpy(float) + rng.normal(0.0, 0.35, n_local)
    explore_temp = rng.uniform(
        max(min_pyrolysis_temp_c, temp_q05 - 100.0),
        min(max_pyrolysis_temp_c, temp_q95 + 150.0),
        n_explore,
    )
    explore_time = rng.uniform(
        max(min_pyrolysis_time_h, time_q05 - 0.25),
        min(max_pyrolysis_time_h, time_q95 + 1.0),
        n_explore,
    )

    compositions = np.vstack([local_comp, explore_comp])
    temperatures = np.concatenate([local_temp, explore_temp])
    times = np.concatenate([local_time, explore_time])
    temperatures = np.clip(temperatures, min_pyrolysis_temp_c, max_pyrolysis_temp_c)
    times = np.clip(times, min_pyrolysis_time_h, max_pyrolysis_time_h)

    candidates = pd.DataFrame(compositions, columns=COMPOSITION_COLUMNS)
    candidates["pyrolysis_temp_c"] = temperatures
    candidates["pyrolysis_time_h"] = times
    candidates["cycling_numbers"] = int(cycle_number)
    candidates["polymer"] = "SDL candidate; precursor route requires review"
    candidates["pyrolysis_atmosphere"] = "inert"
    candidates["surface_area_m2_g"] = np.nan
    candidates["candidate_origin"] = ["local_perturbation"] * n_local + ["exploratory"] * n_explore

    candidates[COMPOSITION_COLUMNS] = candidates[COMPOSITION_COLUMNS].round(3)
    candidates["pyrolysis_temp_c"] = candidates["pyrolysis_temp_c"].round(1)
    candidates["pyrolysis_time_h"] = candidates["pyrolysis_time_h"].round(2)
    candidates.insert(0, "candidate_id", [f"SDL-{i:05d}" for i in range(1, len(candidates) + 1)])
    return candidates


def _predict_model(model_bundle: dict[str, Any], engineered: pd.DataFrame) -> np.ndarray:
    columns = list(model_bundle.get("feature_columns", []))
    matrix = engineered.copy()
    for col in columns:
        if col not in matrix.columns:
            matrix[col] = np.nan
    return np.asarray(model_bundle["model"].predict(matrix[columns]), dtype=float)


def predict_design_performance(candidates: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Predict performance after explicitly excluding measured characterization."""
    production_input = candidates.drop(
        columns=[column for column in CHARACTERIZATION_FIELDS if column in candidates],
        errors="ignore",
    )
    engineered = prepare_features(production_input)
    result = candidates.copy()

    qrev = np.clip(_predict_model(bundle["first_reversible"], engineered), 0.0, None)
    ce = np.clip(_predict_model(bundle["ce_design"], engineered), 1.0, 99.9)
    qcycled_raw = np.clip(_predict_model(bundle["stable_design"], engineered), 0.0, None)
    qcycled = np.minimum(qcycled_raw, qrev)
    qirrev = qrev * (100.0 / ce - 1.0)
    retention = np.divide(
        100.0 * qcycled,
        qrev,
        out=np.full_like(qrev, np.nan),
        where=qrev > 0,
    )

    result["predicted_qrev_mah_g"] = qrev
    result["predicted_qirrev_mah_g"] = qirrev
    result["predicted_ce_pct"] = ce
    result["predicted_qcycled_mah_g"] = qcycled
    result["predicted_apparent_retention_pct"] = retention
    return result


def _expected_improvement(mean: np.ndarray, std: np.ndarray, baseline: float, xi: float = 5.0) -> np.ndarray:
    std = np.asarray(std, dtype=float)
    improvement = np.asarray(mean, dtype=float) - baseline - xi
    z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 1e-12)
    cdf = 0.5 * (1.0 + np.array([erf(float(value) / sqrt(2.0)) for value in z]))
    pdf = np.exp(-0.5 * z**2) / sqrt(2.0 * pi)
    expected = improvement * cdf + std * pdf
    expected[std <= 1e-12] = 0.0
    return np.clip(expected, 0.0, None)


def add_gaussian_process_acquisition(
    candidates: pd.DataFrame,
    bundle: dict[str, Any],
    random_state: int = 42,
    reference_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a fixed-kernel GP acquisition layer without changing production models."""
    reference = _numeric_reference(bundle, reference_override=reference_override)
    training = reference.dropna(subset=[*DESIGN_COLUMNS, TARGET]).copy()
    if len(training) < 15:
        raise ValueError("At least 15 complete public analog rows are required for the GP acquisition model.")

    scaler = StandardScaler()
    train_x = scaler.fit_transform(training[DESIGN_COLUMNS])
    candidate_x = scaler.transform(candidates[DESIGN_COLUMNS])
    train_y = training[TARGET].to_numpy(float)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(length_scale=np.ones(len(DESIGN_COLUMNS)), length_scale_bounds="fixed", nu=2.5)
        + WhiteKernel(noise_level=0.08, noise_level_bounds="fixed")
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.03,
        normalize_y=True,
        optimizer=None,
        random_state=random_state,
    )
    gp.fit(train_x, train_y)
    mean, std = gp.predict(candidate_x, return_std=True)
    baseline = float(np.quantile(train_y, 0.90))

    result = candidates.copy()
    result["gp_qrev_mean_mah_g"] = mean
    result["gp_qrev_std_mah_g"] = std
    result["expected_improvement_mah_g"] = _expected_improvement(mean, std, baseline)
    result["gp_improvement_baseline_mah_g"] = baseline
    return result


def _nearest_distances(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    ref_values = reference[columns].apply(pd.to_numeric, errors="coerce")
    medians = ref_values.median()
    ref_values = ref_values.fillna(medians).to_numpy(float)
    query_values = query[columns].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(float)
    scale = np.nanstd(ref_values, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    ref_scaled = (ref_values - np.nanmedian(ref_values, axis=0)) / scale
    query_scaled = (query_values - np.nanmedian(ref_values, axis=0)) / scale

    distances = np.empty(len(query_scaled), dtype=float)
    indices = np.empty(len(query_scaled), dtype=int)
    for start in range(0, len(query_scaled), 500):
        stop = min(start + 500, len(query_scaled))
        diff = query_scaled[start:stop, None, :] - ref_scaled[None, :, :]
        block = np.sqrt(np.sum(diff**2, axis=2))
        indices[start:stop] = np.argmin(block, axis=1)
        distances[start:stop] = block[np.arange(stop - start), indices[start:stop]]
    return distances, indices


def _standardized_design_arrays(
    query: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    ref_values = reference[DESIGN_COLUMNS].apply(pd.to_numeric, errors="coerce")
    medians = ref_values.median()
    ref_values = ref_values.fillna(medians)
    query_values = query[DESIGN_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(medians)
    scaler = StandardScaler()
    return scaler.fit_transform(ref_values), scaler.transform(query_values)


def _empirical_cdf(reference_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference_values, dtype=float))
    return np.searchsorted(sorted_reference, query_values, side="right") / len(sorted_reference)


def add_domain_and_literature_context(
    candidates: pd.DataFrame,
    bundle: dict[str, Any],
    novelty_method: str = "legacy",
    kde_bandwidth: float = 1.0,
    reference_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add selectable domain scores and public literature context.

    All novelty methods use only pre-synthesis design variables. Optional
    characterization measurements are intentionally excluded. ``legacy``
    remains the default and preserves the previous standardized nearest-neighbor
    behavior and canonical output columns.
    """
    if novelty_method not in NOVELTY_METHODS:
        raise ValueError(
            f"Unsupported novelty_method {novelty_method!r}; choose from {sorted(NOVELTY_METHODS)}."
        )
    if kde_bandwidth <= 0:
        raise ValueError("kde_bandwidth must be positive.")

    scoring_reference = _numeric_reference(bundle, reference_override=reference_override)
    scoring_reference = scoring_reference.dropna(subset=DESIGN_COLUMNS).reset_index(drop=True)
    if len(scoring_reference) < 3:
        raise ValueError("At least three complete reference rows are required for novelty scoring.")
    literature_reference = _numeric_reference(bundle)

    design_distance, _ = _nearest_distances(candidates, scoring_reference, DESIGN_COLUMNS)
    composition_distance, composition_idx = _nearest_distances(
        candidates, literature_reference, COMPOSITION_COLUMNS
    )
    reference_scaled, candidate_scaled = _standardized_design_arrays(candidates, scoring_reference)

    covariance = LedoitWolf().fit(reference_scaled)
    reference_delta = reference_scaled - covariance.location_
    candidate_delta = candidate_scaled - covariance.location_
    reference_mahalanobis = np.sqrt(
        np.maximum(
            np.einsum("ij,jk,ik->i", reference_delta, covariance.precision_, reference_delta),
            0.0,
        )
    )
    candidate_mahalanobis = np.sqrt(
        np.maximum(
            np.einsum("ij,jk,ik->i", candidate_delta, covariance.precision_, candidate_delta),
            0.0,
        )
    )
    mahalanobis_novelty = _empirical_cdf(reference_mahalanobis, candidate_mahalanobis)

    density_model = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth)
    density_model.fit(reference_scaled)
    reference_log_density = density_model.score_samples(reference_scaled)
    candidate_log_density = density_model.score_samples(candidate_scaled)
    sorted_density = np.sort(reference_log_density)
    kde_novelty = 1.0 - (
        np.searchsorted(sorted_density, candidate_log_density, side="left") / len(sorted_density)
    )
    kde_novelty = np.clip(kde_novelty, 0.0, 1.0)

    legacy_confidence = 100.0 / (1.0 + design_distance)
    legacy_novelty = design_distance / (1.0 + design_distance)
    hybrid_novelty = 0.5 * mahalanobis_novelty + 0.5 * kde_novelty
    method_scores = {
        "legacy": (legacy_novelty, legacy_confidence),
        "mahalanobis": (mahalanobis_novelty, 100.0 * (1.0 - mahalanobis_novelty)),
        "kde": (kde_novelty, 100.0 * (1.0 - kde_novelty)),
        "hybrid": (hybrid_novelty, 100.0 * (1.0 - hybrid_novelty)),
    }
    selected_novelty, selected_confidence = method_scores[novelty_method]

    result = candidates.copy()
    result["design_domain_distance"] = design_distance
    result["legacy_design_domain_distance"] = design_distance
    result["legacy_domain_confidence_pct"] = legacy_confidence
    result["legacy_novelty_score"] = legacy_novelty
    result["mahalanobis_distance"] = candidate_mahalanobis
    result["mahalanobis_domain_confidence_pct"] = 100.0 * (1.0 - mahalanobis_novelty)
    result["mahalanobis_novelty_score"] = mahalanobis_novelty
    result["kde_log_density"] = candidate_log_density
    result["kde_domain_confidence_pct"] = 100.0 * (1.0 - kde_novelty)
    result["kde_novelty_score"] = kde_novelty
    result["hybrid_domain_confidence_pct"] = 100.0 * (1.0 - hybrid_novelty)
    result["hybrid_novelty_score"] = hybrid_novelty
    result["novelty_method"] = novelty_method
    result["domain_confidence_pct"] = selected_confidence
    result["novelty_score"] = selected_novelty
    result["composition_match_pct"] = 100.0 / (1.0 + composition_distance)

    closest = literature_reference.iloc[composition_idx].reset_index(drop=True)
    mappings = {
        "polymer": "closest_literature_polymer",
        "polymer_family_broad": "closest_precursor_family",
        "reference": "closest_reference",
        "doi": "closest_doi",
    }
    for source, target in mappings.items():
        result[target] = closest[source].fillna("").astype(str).to_numpy() if source in closest.columns else ""
    result["closest_doi_url"] = result["closest_doi"].apply(
        lambda value: "" if not str(value).strip() else f"https://doi.org/{str(value).strip().replace('https://doi.org/', '')}"
    )

    stats = bundle.get(PUBLIC_RANGE_STATS_KEY, {})
    common_flags = []
    full_flags = []
    for _, row in result.iterrows():
        common = True
        full = True
        for col in DESIGN_COLUMNS:
            if col not in stats:
                continue
            value = float(row[col])
            common = common and float(stats[col]["q05"]) <= value <= float(stats[col]["q95"])
            full = full and float(stats[col]["min"]) <= value <= float(stats[col]["max"])
        common_flags.append(common)
        full_flags.append(full)
    result["inside_common_training_domain"] = common_flags
    result["inside_full_training_domain"] = full_flags
    return result


def _robust_unit_scale(values: pd.Series, higher_is_better: bool = True) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(len(array))
    low, high = np.quantile(finite, [0.05, 0.95])
    if high - low <= 1e-12:
        scaled = np.full(len(array), 0.5)
    else:
        scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return scaled if higher_is_better else 1.0 - scaled


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    efficient = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        if not efficient[index]:
            continue
        dominates = np.all(values >= values[index], axis=1) & np.any(values > values[index], axis=1)
        dominates[index] = False
        if dominates.any():
            efficient[index] = False
    return efficient


def rank_candidate_space(
    candidates: pd.DataFrame,
    exploration_weight: float = 0.35,
) -> pd.DataFrame:
    """Rank candidate experiments using performance, exploration, and domain confidence."""
    if not 0.0 <= exploration_weight <= 1.0:
        raise ValueError("exploration_weight must be between 0 and 1.")

    result = candidates.copy()
    qrev_score = _robust_unit_scale(result["predicted_qrev_mah_g"])
    ce_score = _robust_unit_scale(result["predicted_ce_pct"])
    qcycled_score = _robust_unit_scale(result["predicted_qcycled_mah_g"])
    qirrev_score = _robust_unit_scale(result["predicted_qirrev_mah_g"], higher_is_better=False)

    performance = 0.35 * qrev_score + 0.20 * ce_score + 0.30 * qcycled_score + 0.15 * qirrev_score
    uncertainty = _robust_unit_scale(result["gp_qrev_std_mah_g"])
    improvement = _robust_unit_scale(result["expected_improvement_mah_g"])
    novelty = np.clip(pd.to_numeric(result["novelty_score"], errors="coerce").fillna(0).to_numpy(), 0.0, 1.0)
    exploration = 0.45 * uncertainty + 0.35 * improvement + 0.20 * novelty

    confidence = np.clip(
        pd.to_numeric(result["domain_confidence_pct"], errors="coerce").fillna(0).to_numpy() / 100.0,
        0.0,
        1.0,
    )
    acquisition = ((1.0 - exploration_weight) * performance + exploration_weight * exploration)
    acquisition *= 0.75 + 0.25 * confidence

    result["performance_score_pct"] = 100.0 * performance
    result["exploration_score_pct"] = 100.0 * exploration
    result["acquisition_score_pct"] = 100.0 * acquisition
    objectives = np.column_stack([qrev_score, ce_score, qcycled_score, qirrev_score])
    result["pareto_front"] = _pareto_mask(objectives)
    return result.sort_values(
        ["pareto_front", "acquisition_score_pct", "domain_confidence_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def select_diverse_recommendations(
    ranked: pd.DataFrame,
    n_suggestions: int = 10,
    minimum_distance: float = 0.45,
) -> pd.DataFrame:
    """Greedily select high-ranking candidates that are not near-duplicates."""
    if n_suggestions < 1:
        raise ValueError("n_suggestions must be positive.")
    feature_values = ranked[DESIGN_COLUMNS].apply(pd.to_numeric, errors="coerce")
    feature_values = feature_values.fillna(feature_values.median()).to_numpy(float)
    scale = np.std(feature_values, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    scaled = (feature_values - np.median(feature_values, axis=0)) / scale

    selected: list[int] = []
    for index in range(len(ranked)):
        if not selected:
            selected.append(index)
        else:
            distances = np.sqrt(np.sum((scaled[selected] - scaled[index]) ** 2, axis=1))
            if np.all(distances >= minimum_distance):
                selected.append(index)
        if len(selected) >= n_suggestions:
            break

    if len(selected) < n_suggestions:
        selected.extend(index for index in range(len(ranked)) if index not in selected)
        selected = selected[:n_suggestions]

    recommendations = ranked.iloc[selected].copy().reset_index(drop=True)
    recommendations.insert(0, "recommendation_rank", np.arange(1, len(recommendations) + 1))
    return recommendations


def recommend_next_experiments(
    bundle: dict[str, Any],
    n_candidates: int = 2500,
    n_suggestions: int = 10,
    exploration_fraction: float = 0.40,
    exploration_weight: float = 0.35,
    minimum_distance: float = 0.45,
    cycle_number: int = 100,
    min_si_wt_pct: float = 5.0,
    min_c_wt_pct: float = 5.0,
    max_n_wt_pct: float = 26.0,
    min_pyrolysis_temp_c: float = 800.0,
    max_pyrolysis_temp_c: float = 1400.0,
    min_pyrolysis_time_h: float = 0.5,
    max_pyrolysis_time_h: float = 6.0,
    random_state: int = 42,
    novelty_method: str = "legacy",
    kde_bandwidth: float = 1.0,
    acquisition_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run candidate ranking while keeping deployed prediction models fixed.

    ``acquisition_reference`` can contain newly measured outcomes, but those
    rows affect only local candidate generation, GP uncertainty/expected
    improvement, and novelty scoring. Production Qrev, CE, and Qcycled
    predictions always come from the unchanged design-stage bundle.
    """
    candidates = generate_candidate_space(
        bundle,
        n_candidates=n_candidates,
        exploration_fraction=exploration_fraction,
        cycle_number=cycle_number,
        min_si_wt_pct=min_si_wt_pct,
        min_c_wt_pct=min_c_wt_pct,
        max_n_wt_pct=max_n_wt_pct,
        min_pyrolysis_temp_c=min_pyrolysis_temp_c,
        max_pyrolysis_temp_c=max_pyrolysis_temp_c,
        min_pyrolysis_time_h=min_pyrolysis_time_h,
        max_pyrolysis_time_h=max_pyrolysis_time_h,
        random_state=random_state,
        reference_override=acquisition_reference,
    )
    candidates = predict_design_performance(candidates, bundle)
    candidates = add_gaussian_process_acquisition(
        candidates,
        bundle,
        random_state=random_state,
        reference_override=acquisition_reference,
    )
    candidates = add_domain_and_literature_context(
        candidates,
        bundle,
        novelty_method=novelty_method,
        kde_bandwidth=kde_bandwidth,
        reference_override=acquisition_reference,
    )
    ranked = rank_candidate_space(candidates, exploration_weight=exploration_weight)
    selected = select_diverse_recommendations(
        ranked,
        n_suggestions=n_suggestions,
        minimum_distance=minimum_distance,
    )
    return selected, ranked


def build_experiment_manifest(
    recommendations: pd.DataFrame,
    bundle_path: str | Path,
    candidate_count: int,
    exploration_fraction: float,
    exploration_weight: float,
    cycle_number: int,
    replicate_count: int = 3,
    acquisition_method: str = "fixed_kernel_gp_expected_improvement",
    novelty_method: str = "legacy",
    input_data_hash: str | None = None,
    measurement_provenance: dict[str, Any] | None = None,
    failed_experiments: list[dict[str, Any]] | None = None,
    replicate_statistics: list[dict[str, Any]] | None = None,
    reference_row_counts: dict[str, int] | None = None,
    parent_manifest: str | None = None,
) -> dict[str, Any]:
    """Build an SDL/ELN manifest while preserving the human-review boundary."""
    bundle_path = Path(bundle_path)
    experiments = []
    for _, row in recommendations.iterrows():
        acquisition = {
            "method": acquisition_method,
            "novelty_method": str(row.get("novelty_method", novelty_method)),
            "score_pct": round(float(row["acquisition_score_pct"]), 2),
            "performance_score_pct": round(float(row["performance_score_pct"]), 2),
            "exploration_score_pct": round(float(row["exploration_score_pct"]), 2),
            "gp_qrev_std_mah_g": round(float(row["gp_qrev_std_mah_g"]), 2),
            "expected_improvement_mah_g": round(float(row["expected_improvement_mah_g"]), 2),
            "domain_confidence_pct": round(float(row["domain_confidence_pct"]), 2),
            "novelty_score": round(float(row["novelty_score"]), 6),
            "pareto_front": bool(row["pareto_front"]),
        }
        optional_acquisition_columns = {
            "legacy_design_domain_distance": "legacy_design_domain_distance",
            "legacy_domain_confidence_pct": "legacy_domain_confidence_pct",
            "legacy_novelty_score": "legacy_novelty_score",
            "mahalanobis_distance": "mahalanobis_distance",
            "mahalanobis_domain_confidence_pct": "mahalanobis_domain_confidence_pct",
            "mahalanobis_novelty_score": "mahalanobis_novelty_score",
            "kde_log_density": "kde_log_density",
            "kde_domain_confidence_pct": "kde_domain_confidence_pct",
            "kde_novelty_score": "kde_novelty_score",
            "hybrid_domain_confidence_pct": "hybrid_domain_confidence_pct",
            "hybrid_novelty_score": "hybrid_novelty_score",
        }
        for output_name, column in optional_acquisition_columns.items():
            value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors="coerce").iloc[0]
            if np.isfinite(value):
                acquisition[output_name] = round(float(value), 6)

        experiments.append(
            {
                "experiment_id": str(row["candidate_id"]),
                "recommendation_rank": int(row["recommendation_rank"]),
                "status": "proposed_human_review_required",
                "objective": "Multi-objective SiOC/SiOCN anode screening",
                "target_composition_wt_pct": {
                    "si": round(float(row["si_wt_pct"]), 3),
                    "c": round(float(row["c_wt_pct"]), 3),
                    "o": round(float(row["o_wt_pct"]), 3),
                    "n": round(float(row["n_wt_pct"]), 3),
                    "sum": round(float(row[COMPOSITION_COLUMNS].sum()), 3),
                },
                "process": {
                    "pyrolysis_temperature_c": round(float(row["pyrolysis_temp_c"]), 1),
                    "pyrolysis_time_h": round(float(row["pyrolysis_time_h"]), 2),
                    "pyrolysis_atmosphere": "inert",
                    "cycle_number_for_prediction": int(row["cycling_numbers"]),
                },
                "predicted_performance": {
                    "qrev_mah_g": round(float(row["predicted_qrev_mah_g"]), 2),
                    "qirrev_mah_g": round(float(row["predicted_qirrev_mah_g"]), 2),
                    "ce_pct": round(float(row["predicted_ce_pct"]), 2),
                    "qcycled_mah_g": round(float(row["predicted_qcycled_mah_g"]), 2),
                    "apparent_retention_pct": round(float(row["predicted_apparent_retention_pct"]), 2),
                },
                "acquisition": acquisition,
                "literature_context": {
                    "role": "composition-nearest public context; not a copied recipe or prediction",
                    "composition_match_pct": round(float(row["composition_match_pct"]), 2),
                    "polymer": str(row.get("closest_literature_polymer", "")),
                    "precursor_family": str(row.get("closest_precursor_family", "")),
                    "reference": str(row.get("closest_reference", "")),
                    "doi_url": str(row.get("closest_doi_url", "")),
                },
                "automation_plan": {
                    "execution_level": "high-level SDL plan; hardware mapping requires local validation",
                    "steps": [
                        {"operation": "select_and_validate_precursor_route", "mode": "human_review"},
                        {"operation": "prepare_precursor_solution", "mode": "SDL_ready"},
                        {"operation": "dose_mix_heat_or_crosslink", "mode": "SDL_ready"},
                        {"operation": "dry_and_transfer", "mode": "semi_automated"},
                        {"operation": "inert_pyrolysis", "mode": "external_furnace_module"},
                        {"operation": "characterize_material", "mode": "automated_or_at_line"},
                        {"operation": "fabricate_and_test_electrode", "mode": "external_battery_workflow"},
                        {"operation": "upload_results_and_update_model", "mode": "ELN_ML_closed_loop"},
                    ],
                },
                "characterization_plan": DEFAULT_MEASUREMENT_PLAN,
                "reproducibility_plan": {
                    "replicate_batches": int(replicate_count),
                    "report_batch_mean_and_standard_deviation": True,
                    "record_failed_or_deviating_runs": True,
                },
                "provenance": {
                    "model_bundle": bundle_path.as_posix(),
                    "model_policy": "design-stage models only; no measured electrochemical target is used as an input",
                    "acquisition_reference_policy": (
                        "measured Qrev may update only the GP acquisition and novelty reference layers"
                    ),
                    "candidate_origin": str(row["candidate_origin"]),
                },
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "SiOC/SiOCN SDL next-experiment recommendation",
        "model_bundle": bundle_path.as_posix(),
        "acquisition_method": acquisition_method,
        "novelty_method": novelty_method,
        "selection": {
            "candidate_count": int(candidate_count),
            "suggestion_count": int(len(recommendations)),
            "exploration_fraction": float(exploration_fraction),
            "exploration_weight": float(exploration_weight),
            "cycle_number": int(cycle_number),
            "acquisition_method": acquisition_method,
            "novelty_method": novelty_method,
        },
        "experiments": experiments,
    }
    if input_data_hash:
        manifest["input_data_hash"] = str(input_data_hash)
    if measurement_provenance is not None:
        manifest["measurement_provenance"] = measurement_provenance
    if failed_experiments is not None:
        manifest["failed_experiments"] = failed_experiments
    if replicate_statistics is not None:
        manifest["replicate_statistics"] = replicate_statistics
    if reference_row_counts is not None:
        manifest["reference_row_counts"] = {
            str(key): int(value) for key, value in reference_row_counts.items()
        }
    if parent_manifest:
        manifest["parent_manifest"] = str(parent_manifest)
    return manifest
