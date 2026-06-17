"""Create the closed-loop SDL retraining demonstration notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "07_sdl_closed_loop_retraining_demo.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip() + "\n",
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def main() -> None:
    cells = [
        md(
            """
            # Closed-Loop SDL Measurement Update Demo

            This notebook demonstrates a complete, human-reviewed materials acceleration loop:

            **recommend -> synthesize -> characterize -> measure -> update acquisition -> recommend**

            The mock measurements update only the Gaussian-process acquisition and novelty/reference
            layers. The deployed design-stage Qrev, CE, and Qcycled models are loaded unchanged.
            Characterization descriptors are stored for provenance and future model development, but
            they are not production prediction features.
            """
        ),
        md(
            """
            ## Safety and model boundary

            Every generated experiment retains
            `status = proposed_human_review_required`. This notebook does not control hardware,
            approve precursor chemistry, or retrain the deployed capacity models.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import subprocess
            import sys

            import pandas as pd

            cwd = Path.cwd().resolve()
            PROJECT_ROOT = next(
                (
                    root
                    for root in [cwd, cwd.parent, cwd.parent.parent]
                    if (root / "models" / "sioc_app_target_models.joblib").exists()
                ),
                None,
            )
            if PROJECT_ROOT is None:
                raise FileNotFoundError("Could not locate the project root.")
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.sdl import (
                CHARACTERIZATION_FIELDS,
                build_experiment_manifest,
                load_sdl_bundle,
                recommend_next_experiments,
            )

            MODEL_PATH = PROJECT_ROOT / "models" / "sioc_app_target_models.joblib"
            RUN_DIR = PROJECT_ROOT / "reports" / "sdl_runs" / "closed_loop_notebook_demo"
            INITIAL_DIR = RUN_DIR / "initial"
            UPDATED_DIR = RUN_DIR / "updated"
            INITIAL_DIR.mkdir(parents=True, exist_ok=True)
            UPDATED_DIR.mkdir(parents=True, exist_ok=True)
            """
        ),
        md("## 1. Generate the initial proposed experiment batch"),
        code(
            """
            bundle = load_sdl_bundle(MODEL_PATH)
            initial_recommendations, initial_ranked = recommend_next_experiments(
                bundle,
                n_candidates=350,
                n_suggestions=6,
                novelty_method="legacy",
                random_state=17,
            )
            initial_manifest = build_experiment_manifest(
                initial_recommendations,
                bundle_path=MODEL_PATH,
                candidate_count=len(initial_ranked),
                exploration_fraction=0.40,
                exploration_weight=0.35,
                cycle_number=100,
                novelty_method="legacy",
            )

            INITIAL_RECOMMENDATIONS = INITIAL_DIR / "next_experiments.csv"
            INITIAL_MANIFEST = INITIAL_DIR / "experiment_manifest.json"
            initial_recommendations.to_csv(INITIAL_RECOMMENDATIONS, index=False)
            INITIAL_MANIFEST.write_text(json.dumps(initial_manifest, indent=2), encoding="utf-8")

            display(
                initial_recommendations[
                    [
                        "recommendation_rank",
                        "candidate_id",
                        "predicted_qrev_mah_g",
                        "gp_qrev_std_mah_g",
                        "domain_confidence_pct",
                    ]
                ].round(2)
            )
            """
        ),
        md(
            """
            ## 2. Create mock measured outcomes

            Two experiments have replicate measurements, one replicate is explicitly failed, and
            the optional characterization fields are populated. A real workflow would export this
            table from the ELN/LIMS after human review and measurement quality control.
            """
        ),
        code(
            """
            first = initial_recommendations.iloc[0]
            second = initial_recommendations.iloc[1]
            third = initial_recommendations.iloc[2]

            mock_measurements = pd.DataFrame(
                [
                    {
                        "experiment_id": first["candidate_id"],
                        "replicate_id": "R1",
                        "measurement_status": "completed",
                        "measurement_quality_flag": "pass",
                        "measured_qrev_mah_g": first["predicted_qrev_mah_g"] * 0.96,
                        "measured_ce_pct": first["predicted_ce_pct"] * 0.99,
                        "crystallinity_pct": 18.0,
                        "grain_size_nm": 8.5,
                        "bet_surface_area_m2_g": 310.0,
                        "raman_d_g_ratio": 1.12,
                        "pore_volume_cm3_g": 0.38,
                    },
                    {
                        "experiment_id": first["candidate_id"],
                        "replicate_id": "R2",
                        "measurement_status": "completed",
                        "measurement_quality_flag": "warning",
                        "measured_qrev_mah_g": first["predicted_qrev_mah_g"] * 0.92,
                        "measured_ce_pct": first["predicted_ce_pct"] * 0.98,
                        "crystallinity_pct": 20.0,
                        "grain_size_nm": 9.1,
                        "bet_surface_area_m2_g": 295.0,
                        "raman_d_g_ratio": 1.08,
                        "pore_volume_cm3_g": 0.35,
                    },
                    {
                        "experiment_id": second["candidate_id"],
                        "replicate_id": "R1",
                        "measurement_status": "completed",
                        "measurement_quality_flag": "pass",
                        "measured_qrev_mah_g": second["predicted_qrev_mah_g"] * 1.04,
                        "measured_ce_pct": second["predicted_ce_pct"],
                        "crystallinity_pct": 14.0,
                        "grain_size_nm": 6.4,
                        "bet_surface_area_m2_g": 420.0,
                        "raman_d_g_ratio": 1.21,
                        "pore_volume_cm3_g": 0.46,
                    },
                    {
                        "experiment_id": third["candidate_id"],
                        "replicate_id": "R1",
                        "measurement_status": "failed",
                        "measurement_quality_flag": "fail",
                        "failure_reason": "Cell sealing failure; no electrochemical result accepted.",
                    },
                ]
            )
            MEASUREMENTS = RUN_DIR / "mock_measurements.csv"
            mock_measurements.to_csv(MEASUREMENTS, index=False)
            display(mock_measurements)
            print("Stored characterization fields:", CHARACTERIZATION_FIELDS)
            """
        ),
        md("## 3. Update the GP/reference layer and regenerate recommendations"),
        code(
            """
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "update_sdl_with_measurements.py"),
                "--prior-manifest",
                str(INITIAL_MANIFEST),
                "--recommendations",
                str(INITIAL_RECOMMENDATIONS),
                "--measurements",
                str(MEASUREMENTS),
                "--model-bundle",
                str(MODEL_PATH),
                "--history-path",
                str(RUN_DIR / "measurement_history.csv"),
                "--output-dir",
                str(UPDATED_DIR),
                "--n-candidates",
                "350",
                "--n-suggestions",
                "6",
                "--novelty-method",
                "hybrid",
                "--seed",
                "23",
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            print(completed.stdout)
            """
        ),
        md("## 4. Inspect provenance, failures, and the new ranked batch"),
        code(
            """
            updated_recommendations = pd.read_csv(UPDATED_DIR / "next_experiments.csv")
            updated_manifest = json.loads(
                (UPDATED_DIR / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            run_metadata = json.loads(
                (UPDATED_DIR / "run_metadata.json").read_text(encoding="utf-8")
            )

            display(
                updated_recommendations[
                    [
                        "recommendation_rank",
                        "candidate_id",
                        "predicted_qrev_mah_g",
                        "gp_qrev_std_mah_g",
                        "novelty_method",
                        "novelty_score",
                        "domain_confidence_pct",
                    ]
                ].round(3)
            )
            display(pd.DataFrame(updated_manifest["replicate_statistics"]))
            display(pd.DataFrame(updated_manifest["failed_experiments"]))

            print("Reference rows:", run_metadata["reference_row_counts"])
            print("Production models updated:", updated_manifest["measurement_provenance"]["production_models_updated"])
            assert updated_manifest["measurement_provenance"]["production_models_updated"] is False
            assert all(
                experiment["status"] == "proposed_human_review_required"
                for experiment in updated_manifest["experiments"]
            )
            """
        ),
        md("## 5. Validate the generated manifest schema"),
        code(
            """
            from jsonschema import Draft202012Validator

            schema = json.loads(
                (PROJECT_ROOT / "schemas" / "sioc_sdl_experiment.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            Draft202012Validator(schema).validate(updated_manifest)
            print("Manifest is valid against schemas/sioc_sdl_experiment.schema.json")
            """
        ),
        md(
            """
            ## Interpretation

            The measured replicates change GP uncertainty, expected improvement, novelty/domain
            estimates, and therefore the acquisition ranking. They do not alter the deployed
            design-stage capacity or efficiency models. This gives the project a genuine iterative
            MAP structure while retaining an explicit model-development boundary until enough
            characterization-linked data exist for a separately validated production model.
            """
        ),
    ]

    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"sdl-closed-loop-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
