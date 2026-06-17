"""Generate SDL-style next-experiment recommendations for SiOC/SiOCN anodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sdl import build_experiment_manifest, load_sdl_bundle, recommend_next_experiments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank proposed SiOC/SiOCN compositions and pyrolysis conditions using "
            "design-stage models, Gaussian-process uncertainty, novelty, and Pareto screening."
        )
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=PROJECT_ROOT / "models" / "sioc_app_target_models.joblib",
    )
    parser.add_argument("--n-candidates", type=int, default=2500)
    parser.add_argument("--n-suggestions", type=int, default=10)
    parser.add_argument("--cycle-number", type=int, default=100)
    parser.add_argument("--min-si-wt-pct", type=float, default=5.0)
    parser.add_argument("--min-c-wt-pct", type=float, default=5.0)
    parser.add_argument("--max-n-wt-pct", type=float, default=26.0)
    parser.add_argument("--min-pyrolysis-temp-c", type=float, default=800.0)
    parser.add_argument("--max-pyrolysis-temp-c", type=float, default=1400.0)
    parser.add_argument("--min-pyrolysis-time-h", type=float, default=0.5)
    parser.add_argument("--max-pyrolysis-time-h", type=float, default=6.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.40)
    parser.add_argument("--exploration-weight", type=float, default=0.35)
    parser.add_argument("--minimum-distance", type=float, default=0.45)
    parser.add_argument(
        "--novelty-method",
        choices=["legacy", "mahalanobis", "kde", "hybrid"],
        default="legacy",
        help="Domain/novelty score used for ranking; legacy preserves previous behavior.",
    )
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=1.0,
        help="Gaussian KDE bandwidth used by kde and hybrid novelty modes.",
    )
    parser.add_argument("--replicate-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "sdl_runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_sdl_bundle(args.model_bundle)
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
    )

    manifest = build_experiment_manifest(
        recommendations,
        bundle_path=args.model_bundle,
        candidate_count=len(ranked),
        exploration_fraction=args.exploration_fraction,
        exploration_weight=args.exploration_weight,
        cycle_number=args.cycle_number,
        replicate_count=args.replicate_count,
        novelty_method=args.novelty_method,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = args.output_dir / "ranked_candidate_space.csv"
    recommendations_path = args.output_dir / "next_experiments.csv"
    manifest_path = args.output_dir / "experiment_manifest.json"

    ranked.to_csv(ranked_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    display_columns = [
        "recommendation_rank",
        "candidate_id",
        "si_wt_pct",
        "c_wt_pct",
        "o_wt_pct",
        "n_wt_pct",
        "pyrolysis_temp_c",
        "pyrolysis_time_h",
        "predicted_qrev_mah_g",
        "predicted_ce_pct",
        "predicted_qcycled_mah_g",
        "gp_qrev_std_mah_g",
        "domain_confidence_pct",
        "acquisition_score_pct",
        "pareto_front",
    ]
    print(recommendations[display_columns].round(2).to_string(index=False))
    print(f"\nSaved ranked candidates: {ranked_path}")
    print(f"Saved next experiments: {recommendations_path}")
    print(f"Saved SDL/ELN manifest: {manifest_path}")
    print(
        "\nSafety boundary: precursor selection, hardware mapping, pyrolysis, electrode fabrication, "
        "and battery testing require expert review before execution."
    )


if __name__ == "__main__":
    main()
