"""Update the SDL acquisition layer with human-reviewed experimental results.

This script never retrains or replaces the deployed Qrev, CE, or Qcycled
prediction models. Measured Qrev values update only the lightweight Gaussian
process acquisition/reference layer used to rank the next proposed experiments.
Optional characterization fields are stored for provenance and future research.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sdl import (  # noqa: E402
    CHARACTERIZATION_FIELDS,
    build_experiment_manifest,
    build_updated_acquisition_reference,
    join_measurements_to_plan,
    load_sdl_bundle,
    recommend_next_experiments,
)


ACQUISITION_METHOD = "fixed_kernel_gp_expected_improvement"


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest for an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_input_hash(paths: list[Path]) -> str:
    """Hash ordered input file names and contents for run provenance."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def manifest_to_plan(manifest: dict[str, Any]) -> pd.DataFrame:
    """Convert prior manifest experiments into immutable planned designs."""
    records = []
    for experiment in manifest.get("experiments", []):
        composition = experiment.get("target_composition_wt_pct", {})
        process = experiment.get("process", {})
        provenance = experiment.get("provenance", {})
        experiment_id = str(experiment.get("experiment_id", "")).strip()
        records.append(
            {
                "experiment_id": experiment_id,
                "candidate_id": experiment_id,
                "recommendation_rank": experiment.get("recommendation_rank"),
                "si_wt_pct": composition.get("si"),
                "c_wt_pct": composition.get("c"),
                "o_wt_pct": composition.get("o"),
                "n_wt_pct": composition.get("n"),
                "pyrolysis_temp_c": process.get("pyrolysis_temperature_c"),
                "pyrolysis_time_h": process.get("pyrolysis_time_h"),
                "candidate_origin": provenance.get("candidate_origin", "prior_manifest"),
            }
        )
    plan = pd.DataFrame(records)
    if plan.empty or (plan["experiment_id"].fillna("").astype(str).str.strip() == "").any():
        raise ValueError("The prior manifest contains no valid experiment identifiers.")
    return plan


def load_prior_plan(
    manifest_path: Path,
    recommendations_path: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load the parent manifest and optional richer recommendation CSV."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_plan = manifest_to_plan(manifest)
    if recommendations_path is None:
        return manifest, manifest_plan

    recommendations = pd.read_csv(recommendations_path)
    if "experiment_id" not in recommendations.columns:
        if "candidate_id" not in recommendations.columns:
            raise ValueError("Recommendations require candidate_id or experiment_id.")
        recommendations["experiment_id"] = recommendations["candidate_id"]
    manifest_ids = set(manifest_plan["experiment_id"].astype(str))
    recommendation_ids = set(recommendations["experiment_id"].astype(str))
    if recommendation_ids != manifest_ids:
        missing = sorted(manifest_ids - recommendation_ids)
        extra = sorted(recommendation_ids - manifest_ids)
        raise ValueError(
            "Recommendation CSV and manifest experiment IDs differ; "
            f"missing={missing}, extra={extra}."
        )
    return manifest, recommendations


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records with nulls for missing values."""
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def append_measurement_history(
    joined: pd.DataFrame,
    history_path: Path,
    snapshot_path: Path,
    source_manifest: Path,
    source_manifest_hash: str,
    measurement_file_hash: str,
    ingested_at_utc: str,
) -> pd.DataFrame:
    """Append validated rows to a durable history and write a versioned snapshot."""
    incoming = joined.copy()
    incoming["ingested_at_utc"] = ingested_at_utc
    incoming["source_manifest"] = source_manifest.as_posix()
    incoming["source_manifest_hash"] = source_manifest_hash
    incoming["measurement_file_hash"] = measurement_file_hash

    if history_path.exists():
        previous = pd.read_csv(history_path)
        history = pd.concat([previous, incoming], ignore_index=True, sort=False)
    else:
        history = incoming

    deduplication_columns = [
        "experiment_id",
        "replicate_id",
        "source_manifest_hash",
        "measurement_file_hash",
    ]
    history = history.drop_duplicates(deduplication_columns, keep="last")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    history.to_csv(snapshot_path, index=False)
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest measured SDL results, update only the GP acquisition/reference layer, "
            "and generate the next human-reviewed recommendation batch."
        )
    )
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--recommendations", type=Path)
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=PROJECT_ROOT / "models" / "sioc_app_target_models.joblib",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "sdl_runs" / "measurement_history.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-candidates", type=int, default=2500)
    parser.add_argument("--n-suggestions", type=int, default=10)
    parser.add_argument("--cycle-number", type=int, default=100)
    parser.add_argument("--exploration-fraction", type=float, default=0.40)
    parser.add_argument("--exploration-weight", type=float, default=0.35)
    parser.add_argument("--minimum-distance", type=float, default=0.45)
    parser.add_argument("--min-si-wt-pct", type=float, default=5.0)
    parser.add_argument("--min-c-wt-pct", type=float, default=5.0)
    parser.add_argument("--max-n-wt-pct", type=float, default=26.0)
    parser.add_argument("--min-pyrolysis-temp-c", type=float, default=800.0)
    parser.add_argument("--max-pyrolysis-temp-c", type=float, default=1400.0)
    parser.add_argument("--min-pyrolysis-time-h", type=float, default=0.5)
    parser.add_argument("--max-pyrolysis-time-h", type=float, default=6.0)
    parser.add_argument(
        "--novelty-method",
        choices=["legacy", "mahalanobis", "kde", "hybrid"],
        default="legacy",
    )
    parser.add_argument("--kde-bandwidth", type=float, default=1.0)
    parser.add_argument("--replicate-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ingested_at_utc = datetime.now(timezone.utc).isoformat()
    output_dir = args.output_dir or (
        PROJECT_ROOT / "reports" / "sdl_runs" / f"closed_loop_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_manifest, prior_plan = load_prior_plan(args.prior_manifest, args.recommendations)
    measurements = pd.read_csv(args.measurements)
    joined = join_measurements_to_plan(measurements, prior_plan)

    manifest_hash = sha256_file(args.prior_manifest)
    measurement_hash = sha256_file(args.measurements)
    hash_inputs = [args.prior_manifest, args.measurements]
    if args.recommendations is not None:
        hash_inputs.append(args.recommendations)
    input_hash = combined_input_hash(hash_inputs)

    snapshot_path = output_dir / f"measurement_history_{timestamp}.csv"
    history = append_measurement_history(
        joined,
        history_path=args.history_path,
        snapshot_path=snapshot_path,
        source_manifest=args.prior_manifest,
        source_manifest_hash=manifest_hash,
        measurement_file_hash=measurement_hash,
        ingested_at_utc=ingested_at_utc,
    )

    bundle = load_sdl_bundle(args.model_bundle)
    current_reference_measurements = history.drop_duplicates(
        ["experiment_id", "replicate_id"], keep="last"
    )
    acquisition_reference, reference_counts, replicate_stats = (
        build_updated_acquisition_reference(bundle, current_reference_measurements)
    )
    recommendations, ranked = recommend_next_experiments(
        bundle,
        n_candidates=args.n_candidates,
        n_suggestions=args.n_suggestions,
        exploration_fraction=args.exploration_fraction,
        exploration_weight=args.exploration_weight,
        minimum_distance=args.minimum_distance,
        cycle_number=args.cycle_number,
        min_si_wt_pct=args.min_si_wt_pct,
        min_c_wt_pct=args.min_c_wt_pct,
        max_n_wt_pct=args.max_n_wt_pct,
        min_pyrolysis_temp_c=args.min_pyrolysis_temp_c,
        max_pyrolysis_temp_c=args.max_pyrolysis_temp_c,
        min_pyrolysis_time_h=args.min_pyrolysis_time_h,
        max_pyrolysis_time_h=args.max_pyrolysis_time_h,
        random_state=args.seed,
        novelty_method=args.novelty_method,
        kde_bandwidth=args.kde_bandwidth,
        acquisition_reference=acquisition_reference,
    )
    candidate_id_map = {
        str(candidate_id): f"SDL-{timestamp}-{index:05d}"
        for index, candidate_id in enumerate(ranked["candidate_id"], start=1)
    }
    ranked["candidate_id"] = ranked["candidate_id"].astype(str).map(candidate_id_map)
    recommendations["candidate_id"] = recommendations["candidate_id"].astype(str).map(
        candidate_id_map
    )

    failed = current_reference_measurements[
        ~current_reference_measurements["measurement_status"].eq("completed")
        | current_reference_measurements["measurement_quality_flag"].eq("fail")
    ].copy()
    failed_records = []
    for _, row in failed.iterrows():
        status = str(row["measurement_status"])
        if status == "completed":
            status = "excluded"
        failed_records.append(
            {
                "experiment_id": str(row["experiment_id"]),
                "replicate_id": str(row["replicate_id"]),
                "measurement_status": status,
                "measurement_quality_flag": str(row["measurement_quality_flag"]),
                "failure_reason": str(row.get("failure_reason", "") or ""),
            }
        )

    measurement_provenance = {
        "measurement_file": args.measurements.as_posix(),
        "measurement_history_file": args.history_path.as_posix(),
        "measurement_file_hash": measurement_hash,
        "ingested_at_utc": ingested_at_utc,
        "accepted_measurement_experiments": reference_counts[
            "accepted_measurement_experiments"
        ],
        "production_models_updated": False,
        "stored_characterization_fields": CHARACTERIZATION_FIELDS,
    }
    manifest = build_experiment_manifest(
        recommendations,
        bundle_path=args.model_bundle,
        candidate_count=len(ranked),
        exploration_fraction=args.exploration_fraction,
        exploration_weight=args.exploration_weight,
        cycle_number=args.cycle_number,
        replicate_count=args.replicate_count,
        acquisition_method=ACQUISITION_METHOD,
        novelty_method=args.novelty_method,
        input_data_hash=input_hash,
        measurement_provenance=measurement_provenance,
        failed_experiments=failed_records,
        replicate_statistics=dataframe_records(replicate_stats),
        reference_row_counts=reference_counts,
        parent_manifest=args.prior_manifest.as_posix(),
    )

    ranked_path = output_dir / "ranked_candidate_space.csv"
    recommendations_path = output_dir / "next_experiments.csv"
    manifest_path = output_dir / "experiment_manifest.json"
    metadata_path = output_dir / "run_metadata.json"
    ranked.to_csv(ranked_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_metadata = {
        "run_timestamp_utc": ingested_at_utc,
        "status": "proposed_human_review_required",
        "model_boundary": (
            "Production prediction models were not retrained; only the GP acquisition "
            "and novelty/reference layers were updated."
        ),
        "acquisition_method": ACQUISITION_METHOD,
        "novelty_method": args.novelty_method,
        "inputs": {
            "prior_manifest": args.prior_manifest.as_posix(),
            "prior_manifest_hash": manifest_hash,
            "recommendations": (
                args.recommendations.as_posix() if args.recommendations is not None else None
            ),
            "measurements": args.measurements.as_posix(),
            "measurement_file_hash": measurement_hash,
            "combined_input_hash": input_hash,
        },
        "outputs": {
            "measurement_history": args.history_path.as_posix(),
            "measurement_history_snapshot": snapshot_path.as_posix(),
            "ranked_candidates": ranked_path.as_posix(),
            "recommendations": recommendations_path.as_posix(),
            "manifest": manifest_path.as_posix(),
        },
        "reference_row_counts": reference_counts,
        "measurement_history_rows": int(len(history)),
        "failed_experiments": failed_records,
        "replicate_statistics": dataframe_records(replicate_stats),
        "parent_workflow": prior_manifest.get("workflow"),
    }
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(recommendations[
        [
            "recommendation_rank",
            "candidate_id",
            "predicted_qrev_mah_g",
            "gp_qrev_std_mah_g",
            "domain_confidence_pct",
            "acquisition_score_pct",
        ]
    ].round(2).to_string(index=False))
    print(f"\nAppended measurement history: {args.history_path}")
    print(f"Saved versioned history snapshot: {snapshot_path}")
    print(f"Saved next recommendations: {recommendations_path}")
    print(f"Saved human-review manifest: {manifest_path}")
    print(f"Saved run metadata: {metadata_path}")
    print(
        "\nBoundary: all outputs remain proposed_human_review_required; "
        "no deployed capacity model was retrained."
    )


if __name__ == "__main__":
    main()
