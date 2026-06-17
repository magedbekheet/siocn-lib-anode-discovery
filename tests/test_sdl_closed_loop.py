"""Lightweight validation tests for the closed-loop SDL extension."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
import numpy as np
import pandas as pd

from src.sdl import (
    CHARACTERIZATION_FIELDS,
    COMPOSITION_COLUMNS,
    DESIGN_COLUMNS,
    add_domain_and_literature_context,
    assert_characterization_not_in_production_features,
    build_experiment_manifest,
    build_updated_acquisition_reference,
    generate_candidate_space,
    join_measurements_to_plan,
    load_sdl_bundle,
    predict_design_performance,
    recommend_next_experiments,
    validate_measurement_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sioc_app_target_models.joblib"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "sioc_sdl_experiment.schema.json"


class SDLClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_sdl_bundle(MODEL_PATH)

    def test_generated_compositions_sum_to_100(self) -> None:
        candidates = generate_candidate_space(self.bundle, n_candidates=40, random_state=4)
        sums = candidates[COMPOSITION_COLUMNS].sum(axis=1)
        self.assertTrue(np.allclose(sums, 100.0, atol=0.01))
        self.assertTrue((candidates[COMPOSITION_COLUMNS] >= 0).all().all())

    def test_measurement_validation_and_allowed_flags(self) -> None:
        valid = pd.DataFrame(
            [
                {
                    "candidate_id": "SDL-1",
                    "replicate_id": "R1",
                    "measurement_status": "completed",
                    "measurement_quality_flag": "pass",
                    "measured_qrev_mah_g": 510.0,
                    "crystallinity_pct": 18.0,
                },
                {
                    "candidate_id": "SDL-2",
                    "replicate_id": "R1",
                    "measurement_status": "failed",
                    "measurement_quality_flag": "fail",
                },
            ]
        )
        normalized = validate_measurement_table(valid)
        self.assertEqual(normalized["experiment_id"].tolist(), ["SDL-1", "SDL-2"])

        with self.assertRaisesRegex(ValueError, "experiment_id or candidate_id"):
            validate_measurement_table(pd.DataFrame({"measured_qrev_mah_g": [500.0]}))
        with self.assertRaisesRegex(ValueError, "Unsupported measurement_status"):
            validate_measurement_table(
                pd.DataFrame(
                    {
                        "experiment_id": ["SDL-1"],
                        "measurement_status": ["unknown"],
                        "measured_qrev_mah_g": [500.0],
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "require measured_qrev"):
            validate_measurement_table(
                pd.DataFrame(
                    {
                        "experiment_id": ["SDL-1"],
                        "measurement_status": ["completed"],
                    }
                )
            )

    def test_all_novelty_methods_return_finite_scores(self) -> None:
        candidates = generate_candidate_space(self.bundle, n_candidates=30, random_state=6)
        for method in ("legacy", "mahalanobis", "kde", "hybrid"):
            scored = add_domain_and_literature_context(
                candidates,
                self.bundle,
                novelty_method=method,
                kde_bandwidth=1.0,
            )
            self.assertTrue(np.isfinite(scored["novelty_score"]).all())
            self.assertTrue(scored["novelty_score"].between(0, 1).all())
            self.assertTrue(scored["domain_confidence_pct"].between(0, 100).all())
            self.assertIn("mahalanobis_distance", scored.columns)
            self.assertIn("kde_log_density", scored.columns)
            self.assertTrue((scored["novelty_method"] == method).all())

    def test_measurements_update_only_acquisition_reference(self) -> None:
        recommendations, _ = recommend_next_experiments(
            self.bundle,
            n_candidates=40,
            n_suggestions=3,
            random_state=8,
        )
        plan = recommendations[["candidate_id", *DESIGN_COLUMNS]].copy()
        measured = pd.DataFrame(
            {
                "experiment_id": [recommendations.iloc[0]["candidate_id"]],
                "replicate_id": ["R1"],
                "measurement_status": ["completed"],
                "measurement_quality_flag": ["pass"],
                "measured_qrev_mah_g": [525.0],
                "bet_surface_area_m2_g": [350.0],
                "raman_d_g_ratio": [1.15],
            }
        )
        joined = join_measurements_to_plan(measured, plan)
        reference, counts, _ = build_updated_acquisition_reference(self.bundle, joined)
        self.assertEqual(counts["accepted_measurement_experiments"], 1)
        self.assertEqual(
            counts["combined_reference_rows"],
            counts["public_reference_rows"] + 1,
        )
        self.assertIn("bet_surface_area_m2_g", reference.columns)

        candidates = generate_candidate_space(self.bundle, n_candidates=12, random_state=12)
        baseline = predict_design_performance(candidates, self.bundle)
        enriched = candidates.copy()
        for column in CHARACTERIZATION_FIELDS:
            enriched[column] = 999.0
        repeated = predict_design_performance(enriched, self.bundle)
        prediction_columns = [
            "predicted_qrev_mah_g",
            "predicted_qirrev_mah_g",
            "predicted_ce_pct",
            "predicted_qcycled_mah_g",
        ]
        np.testing.assert_allclose(
            baseline[prediction_columns].to_numpy(),
            repeated[prediction_columns].to_numpy(),
        )

    def test_characterization_fields_are_not_production_features(self) -> None:
        assert_characterization_not_in_production_features(self.bundle)
        for model_name in ("first_reversible", "ce_design", "stable_design"):
            features = set(self.bundle[model_name]["feature_columns"])
            self.assertFalse(features.intersection(CHARACTERIZATION_FIELDS))

    def test_manifest_validates_against_schema(self) -> None:
        recommendations, ranked = recommend_next_experiments(
            self.bundle,
            n_candidates=40,
            n_suggestions=3,
            novelty_method="hybrid",
            random_state=13,
        )
        manifest = build_experiment_manifest(
            recommendations,
            bundle_path=MODEL_PATH,
            candidate_count=len(ranked),
            exploration_fraction=0.40,
            exploration_weight=0.35,
            cycle_number=100,
            novelty_method="hybrid",
            input_data_hash="a" * 64,
            reference_row_counts={"public_reference_rows": 100},
        )
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(manifest)
        self.assertTrue(
            all(
                experiment["status"] == "proposed_human_review_required"
                for experiment in manifest["experiments"]
            )
        )


if __name__ == "__main__":
    unittest.main()
