"""Create the SDL active-learning extension notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "06_sdl_active_learning_extension.ipynb"


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
            # SDL-Ready SiOC/SiOCN Anode Experiment Recommendation

            This notebook extends the literature-trained SiOC/SiOCN anode models into a transparent
            **next-experiment recommendation prototype** for a self-driving laboratory (SDL) or
            materials acceleration platform (MAP).

            It does not directly control laboratory hardware. It proposes composition and pyrolysis
            candidates, estimates design-stage electrochemical performance, balances exploitation
            and exploration, attaches literature context, and exports a structured manifest for
            expert review, ELN capture, and later hardware integration.

            The workflow is aligned with four SDL principles:

            1. structured candidate generation,
            2. model-guided experiment selection,
            3. reproducible metadata and characterization plans,
            4. feedback of measured results into the next model iteration.
            """
        ),
        md(
            """
            ## Workflow

            ```text
            Public literature analogs + trained design models
                              |
                    Candidate composition space
                              |
              Qrev / CE / Qcycled design predictions
                              |
               GP uncertainty + expected improvement
                              |
               Pareto screening + diversity selection
                              |
                 Human-reviewed experiment manifest
                              |
              Synthesis -> characterization -> ELN -> retraining
            ```

            The current model is trained predominantly on low-current, approximately 0-3 V
            lithium-ion literature data. Recommendations are hypotheses for experimental validation,
            not guaranteed synthesis outcomes.
            """
        ),
        md("## 1. Imports and project paths"),
        code(
            """
            from pathlib import Path
            import json
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
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
                raise FileNotFoundError("Could not find models/sioc_app_target_models.joblib.")
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            MODEL_PATH = PROJECT_ROOT / "models" / "sioc_app_target_models.joblib"
            RUN_DIR = PROJECT_ROOT / "reports" / "sdl_runs" / "notebook_demo"
            RUN_DIR.mkdir(parents=True, exist_ok=True)

            from src.sdl import (
                build_experiment_manifest,
                load_sdl_bundle,
                recommend_next_experiments,
            )

            print("Project root:", PROJECT_ROOT)
            print("Model bundle:", MODEL_PATH)
            """
        ),
        md("## 2. Load the public-safe model and reference layer"),
        code(
            """
            bundle = load_sdl_bundle(MODEL_PATH)
            analogs = bundle["_public_literature_analogs_raw"]

            model_audit = pd.DataFrame(
                [
                    {
                        "target": name,
                        "model": bundle[key]["best_model_name"],
                        "training_rows": bundle[key]["n_training_rows"],
                        "features": ", ".join(bundle[key]["feature_columns"]),
                    }
                    for name, key in [
                        ("Qrev", "first_reversible"),
                        ("CE", "ce_design"),
                        ("Qcycled", "stable_design"),
                    ]
                ]
            )
            display(model_audit)
            print("Public literature analog rows:", len(analogs))
            """
        ),
        md(
            """
            ## 3. Generate and rank candidate experiments

            Candidate generation combines:

            - local perturbations around public literature compositions,
            - exploratory SiOC/SiOCN compositions,
            - final pyrolysis temperature and time,
            - minimum Si and C constraints,
            - an upper N limit matching the current literature domain.

            Ranking combines predicted performance, Gaussian-process uncertainty, expected
            improvement, novelty, domain confidence, Pareto status, and diversity.
            """
        ),
        code(
            """
            N_CANDIDATES = 1500
            N_SUGGESTIONS = 10
            CYCLE_NUMBER = 100
            EXPLORATION_FRACTION = 0.40
            EXPLORATION_WEIGHT = 0.35

            recommendations, ranked = recommend_next_experiments(
                bundle,
                n_candidates=N_CANDIDATES,
                n_suggestions=N_SUGGESTIONS,
                exploration_fraction=EXPLORATION_FRACTION,
                exploration_weight=EXPLORATION_WEIGHT,
                minimum_distance=0.45,
                cycle_number=CYCLE_NUMBER,
                min_si_wt_pct=5.0,
                min_c_wt_pct=5.0,
                max_n_wt_pct=26.0,
                min_pyrolysis_temp_c=800.0,
                max_pyrolysis_temp_c=1400.0,
                min_pyrolysis_time_h=0.5,
                max_pyrolysis_time_h=6.0,
                random_state=42,
            )

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
                "predicted_qirrev_mah_g",
                "predicted_ce_pct",
                "predicted_qcycled_mah_g",
                "gp_qrev_std_mah_g",
                "domain_confidence_pct",
                "acquisition_score_pct",
                "pareto_front",
            ]
            display(recommendations[display_columns].round(2))
            """
        ),
        md("## 4. Performance, uncertainty, and domain plots"),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            scatter = axes[0].scatter(
                ranked["predicted_qrev_mah_g"],
                ranked["predicted_ce_pct"],
                c=ranked["acquisition_score_pct"],
                cmap="viridis",
                alpha=0.45,
                s=22,
            )
            axes[0].scatter(
                recommendations["predicted_qrev_mah_g"],
                recommendations["predicted_ce_pct"],
                facecolors="none",
                edgecolors="#b42318",
                linewidths=1.8,
                s=110,
                label="selected",
            )
            axes[0].set_xlabel("Predicted Qrev (mAh/g)")
            axes[0].set_ylabel("Predicted CE (%)")
            axes[0].set_title("Candidate performance and acquisition score")
            axes[0].legend()
            fig.colorbar(scatter, ax=axes[0], label="Acquisition score (%)")

            axes[1].scatter(
                ranked["domain_confidence_pct"],
                ranked["gp_qrev_std_mah_g"],
                c=ranked["novelty_score"],
                cmap="plasma",
                alpha=0.45,
                s=22,
            )
            axes[1].scatter(
                recommendations["domain_confidence_pct"],
                recommendations["gp_qrev_std_mah_g"],
                facecolors="none",
                edgecolors="#174f59",
                linewidths=1.8,
                s=110,
            )
            axes[1].set_xlabel("Domain confidence (%)")
            axes[1].set_ylabel("GP Qrev uncertainty proxy (mAh/g)")
            axes[1].set_title("Exploration versus training-domain proximity")

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 5. Literature context and precursor review

            Precursor identity is **not** a model input and does not alter the predicted
            electrochemical values. The nearest public literature analog is attached only to help an
            expert select a chemically plausible precursor route for the target composition.
            """
        ),
        code(
            """
            context_columns = [
                "recommendation_rank",
                "candidate_id",
                "composition_match_pct",
                "closest_precursor_family",
                "closest_literature_polymer",
                "closest_reference",
                "closest_doi_url",
            ]
            display(recommendations[context_columns])
            """
        ),
        md("## 6. Export an SDL/ELN-ready experiment manifest"),
        code(
            """
            manifest = build_experiment_manifest(
                recommendations,
                bundle_path=MODEL_PATH,
                candidate_count=len(ranked),
                exploration_fraction=EXPLORATION_FRACTION,
                exploration_weight=EXPLORATION_WEIGHT,
                cycle_number=CYCLE_NUMBER,
                replicate_count=3,
            )

            ranked.to_csv(RUN_DIR / "ranked_candidate_space.csv", index=False)
            recommendations.to_csv(RUN_DIR / "next_experiments.csv", index=False)
            (RUN_DIR / "experiment_manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            print("Saved outputs to:", RUN_DIR)
            print(json.dumps(manifest["experiments"][0], indent=2)[:5000])
            """
        ),
        md(
            """
            ## 7. Laboratory integration boundary

            The manifest is a high-level experimental plan, not a direct hardware command stream.

            **Suitable for SDL automation or orchestration**

            - precursor solution preparation,
            - liquid dosing and mixing,
            - sol-gel/crosslinking screening,
            - heating, stirring, sonication, and washing,
            - metadata capture and ELN upload.

            **Requires additional modules or external workflows**

            - controlled drying,
            - inert-atmosphere high-temperature pyrolysis,
            - solid dosing of powders,
            - electrode formulation and coating,
            - cell assembly and electrochemical testing.

            Every proposed route therefore remains under human review until precursor chemistry,
            hardware compatibility, safety constraints, and analytical methods are validated.
            """
        ),
        md(
            """
            ## 8. Closing the loop

            After each experiment, append measured composition, characterization, batch variability,
            failed runs, and electrochemical results to the ELN/data store. The next model version
            should be trained with DOI/experiment-grouped validation and should explicitly track:

            - model version and feature policy,
            - synthesis and instrument metadata,
            - replicate mean and standard deviation,
            - prediction residuals,
            - domain expansion from newly measured compositions.

            A future Bayesian optimization service can then replace the current fixed candidate pool
            and issue new experiments iteratively as validated measurements become available.
            """
        ),
    ]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"sdl-cell-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
