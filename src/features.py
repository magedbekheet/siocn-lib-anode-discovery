"""Feature engineering utilities for SiOC battery-capacity modeling.

This module is shared by both the modeling notebook and the Streamlit app.
The revised schema includes literature-derived composition, processing,
electrochemical, and cycling-stability descriptors.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

M_SI = 28.0855
M_C = 12.011
M_O = 15.999
M_N = 14.007

TARGET = "reversible_capacity_mah_g"

# Expanded numeric feature schema. The first block is composition/processing
# and is suitable for virtual screening. The later electrochemical/cycling
# fields are useful when literature metadata are known; the app clearly marks
# them as optional because some can only be known after testing.
NUMERIC_FEATURES = [
    # Composition and derived phase proxies
    "si_wt_pct",
    "c_wt_pct",
    "o_wt_pct",
    "n_wt_pct",
    "sio2_phase",
    "sic_phase",
    "free_c_phase",
    "si3n4_phase",
    "c_si_atomic_ratio",
    "o_si_atomic_ratio",
    "n_si_atomic_ratio",
    "c_o_atomic_ratio",
    "heteroatom_wt_pct",
    "non_si_wt_pct",
    # Processing and text-derived synthesis flags
    "pyrolysis_temp_c",
    "pyrolysis_time_h",
    "pyrolysis_thermal_budget",
    "crosslink_temp_c",
    "crosslink_time_h",
    "crosslinking_thermal_budget",
    "crosslinking_used",
    "dvb_modification",
    "phenyl_present",
    "vinyl_present",
    "hf_treated",
    # Microstructure
    "surface_area_m2_g",
    "log_surface_area_m2_g",
    # Electrochemical test conditions
    "first_cycling_current_ma_g",
    "cycling_current_ma_g",
    "voltage_min_v",
    "voltage_max_v",
    "voltage_window_v",
    "solvent_count",
    "additive_count",
    "electrolyte_known",
    # Literature performance/cycling descriptors
    "irreversible_capacity_mah_g",
    "coulombic_efficiency_pct",
    "cycling_reversible_capacity_mah_g",
    "capacity_retention_pct",
    "cycling_numbers",
]

CATEGORICAL_FEATURES = [
    "pyrolysis_atmosphere_group",
    "polymer_family_broad",
    "precursor_family",
]

DISPLAY_FEATURE_GROUPS = {
    "Composition / phase": [
        "si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct",
        "sio2_phase", "sic_phase", "free_c_phase", "si3n4_phase",
        "c_si_atomic_ratio", "o_si_atomic_ratio", "n_si_atomic_ratio",
        "conductive_network_index", "effective_bandgap_proxy_ev",
    ],
    "Processing / microstructure": [
        "pyrolysis_temp_c", "pyrolysis_time_h", "pyrolysis_thermal_budget",
        "crosslink_temp_c", "crosslink_time_h", "surface_area_m2_g",
    ],
    "Electrochemistry / cycling": [
        "first_cycling_current_ma_g", "cycling_current_ma_g",
        "irreversible_capacity_mah_g", "coulombic_efficiency_pct",
        "cycling_reversible_capacity_mah_g", "capacity_retention_pct", "cycling_numbers",
    ],
}

PHASE_PHYSICS_FEATURES = [
    "conductive_network_index",
    "ceramic_confinement_index",
    "effective_bandgap_proxy_ev",
    "phase_weighted_electronegativity",
    "phase_separation_index",
]

PRIMARY_DISCOVERY_NUMERIC_FEATURES = [
    # Compact pre-test feature set for discovering new SiOC/SiOCN compositions.
    # This intentionally avoids polymer identity, sparse structure measurements,
    # protocol conditions, and post-electrochemical performance descriptors.
    "si_wt_pct",
    "c_wt_pct",
    "o_wt_pct",
    "n_wt_pct",
    "pyrolysis_temp_c",
    "pyrolysis_time_h",
]

PROTOCOL_CONDITIONING_FEATURES = [
    # Current density is intentionally excluded from the deployed discovery
    # model because the small literature dataset learns a non-physical trend.
    # Interpret predictions as low-current literature-regime capacities.
]

PRIMARY_DISCOVERY_CATEGORICAL_FEATURES = [
    # Polymer/precursor family is kept for route recommendation, not prediction.
]

OPTIONAL_PROCESS_TEXT_FEATURES = [
    "dvb_modification",
    "dvb_ratio_to_base",
    "dvb_wt_pct_nominal",
    "phenyl_present",
    "vinyl_present",
]

OPTIONAL_STRUCTURE_FEATURES = [
    # Useful only after synthesis/characterization and very sparse in the
    # current literature table, so not part of the primary discovery model.
    "surface_area_m2_g",
]

OPTIONAL_PHASE_PROXY_FEATURES = [
    # Keep as an ablation or ranking aid. In the current grouped-CV runs these
    # did not improve MAE when added to the compact design model.
    "sio2_phase",
    "sic_phase",
    "free_c_phase",
    "si3n4_phase",
    "conductive_network_index",
    "effective_bandgap_proxy_ev",
]

FORBIDDEN_TARGET_PROXY_FEATURES = [
    TARGET,
    "irreversible_capacity_mah_g",
    "coulombic_efficiency_pct",
    "cycling_reversible_capacity_mah_g",
    "capacity_retention_pct",
    "cycling_numbers",
    "cycling_current_ma_g",
]

DISCOVERY_NUMERIC_FEATURES = PRIMARY_DISCOVERY_NUMERIC_FEATURES + PROTOCOL_CONDITIONING_FEATURES
DISCOVERY_CATEGORICAL_FEATURES = PRIMARY_DISCOVERY_CATEGORICAL_FEATURES


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized snake_case column names and known aliases."""
    out = dataframe.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace("%", "pct", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.lower()
    )

    rename_map = {
        "si_wt_pct": "si_wt_pct",
        "c_wt_pct": "c_wt_pct",
        "o_wt_pct": "o_wt_pct",
        "n_wt_pct": "n_wt_pct",
        "pyrolysis_temp_c": "pyrolysis_temp_c",
        "pyrolysis_time_h": "pyrolysis_time_h",
        "crosslink_temp_c": "crosslink_temp_c",
        "crosslink_time_h": "crosslink_time_h",
        "reversible_capacity_mah_g": "reversible_capacity_mah_g",
        "irreversible_capacity_mah_g": "irreversible_capacity_mah_g",
        "cycling_reversible_capacity_mah_g": "cycling_reversible_capacity_mah_g",
        "surface_area_m2_g": "surface_area_m2_g",
        "surface_area_m2g": "surface_area_m2_g",
        "bet_surface_area_m2_g": "surface_area_m2_g",
        "coulombic_efficiency_?": "coulombic_efficiency_pct",
        "coulombic_efficiency_pct": "coulombic_efficiency_pct",
        "capacity_retention_?": "capacity_retention_pct",
        "capacity_retention_pct": "capacity_retention_pct",
        "cycling_numbers": "cycling_numbers",
        "number_of_cycles": "cycling_numbers",
        "cycle_number": "cycling_numbers",
        "cycles": "cycling_numbers",
        "first_cycling_current_ma_g": "first_cycling_current_ma_g",
        "applied_current_ma_g": "first_cycling_current_ma_g",
        "applied_current_mag": "first_cycling_current_ma_g",
        "cycling_current_ma_g": "cycling_current_ma_g",
        "moltage_min": "voltage_min_v",
        "voltage_min": "voltage_min_v",
        "voltage_min_v": "voltage_min_v",
        "voltage_max": "voltage_max_v",
        "voltage_max_v": "voltage_max_v",
        "voltage_window": "voltage_window_v",
        "voltage_window_v": "voltage_window_v",
    }
    out = out.rename(columns={c: rename_map.get(c, c) for c in out.columns})
    return out


def read_sioc_csv(path: str) -> pd.DataFrame:
    """Read CSV robustly; the literature file may contain non-UTF8 characters."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1")


def _to_numeric(dataframe: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = dataframe.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_composition_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add atomic fractions, SiOCN phase proxies, and composition ratios."""
    out = dataframe.copy()
    out = _to_numeric(out, ["si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct"])

    for col in ["si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct"]:
        if col not in out.columns:
            out[col] = 0.0 if col == "n_wt_pct" else np.nan
    out["n_wt_pct"] = out["n_wt_pct"].fillna(0.0)

    out["si_mol"] = out["si_wt_pct"] / M_SI
    out["c_mol"] = out["c_wt_pct"] / M_C
    out["o_mol"] = out["o_wt_pct"] / M_O
    out["n_mol"] = out["n_wt_pct"] / M_N
    mol_total = out[["si_mol", "c_mol", "o_mol", "n_mol"]].sum(axis=1).replace(0, np.nan)

    out["x_si"] = out["si_mol"] / mol_total
    out["y_c"] = out["c_mol"] / mol_total
    out["z_o"] = out["o_mol"] / mol_total
    out["w_n"] = out["n_mol"] / mol_total

    # Idealized SiOCN phase proxies. N is represented as a Si3N4-like proxy first;
    # remaining Si is then partitioned between SiO2 and SiC proxies.
    out["si3n4_phase_raw"] = out["w_n"] / 4
    remaining_si = out["x_si"] - 3 * out["si3n4_phase_raw"]
    out["sio2_phase_raw"] = out["z_o"] / 2
    out["sic_phase_raw"] = remaining_si - out["sio2_phase_raw"]
    out["free_c_phase_raw"] = out["y_c"] - out["sic_phase_raw"]

    for raw_col, clean_col in [
        ("sio2_phase_raw", "sio2_phase"),
        ("sic_phase_raw", "sic_phase"),
        ("free_c_phase_raw", "free_c_phase"),
        ("si3n4_phase_raw", "si3n4_phase"),
    ]:
        out[clean_col] = out[raw_col].clip(lower=0)
        out[f"{clean_col}_negative_flag"] = (out[raw_col] < 0).astype(int)

    out["c_si_atomic_ratio"] = np.where(out["x_si"] > 0, out["y_c"] / out["x_si"], np.nan)
    out["o_si_atomic_ratio"] = np.where(out["x_si"] > 0, out["z_o"] / out["x_si"], np.nan)
    out["n_si_atomic_ratio"] = np.where(out["x_si"] > 0, out["w_n"] / out["x_si"], np.nan)
    out["c_o_atomic_ratio"] = np.where(out["z_o"] > 0, out["y_c"] / out["z_o"], np.nan)
    out["heteroatom_wt_pct"] = out["o_wt_pct"].fillna(0) + out["n_wt_pct"].fillna(0)
    out["non_si_wt_pct"] = 100 - out["si_wt_pct"]
    return out


def add_phase_physics_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add non-leaky phase-physics proxies for SiOC/SiOCN virtual screening.

    These descriptors are heuristic indices derived only from composition-derived
    phase proxies. They should be treated as model features or ranking aids, not
    as direct phase quantification from diffraction/spectroscopy.
    """
    out = dataframe.copy()

    required = ["free_c_phase", "sio2_phase", "sic_phase", "si3n4_phase"]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan

    f_c = out["free_c_phase"]
    f_sio2 = out["sio2_phase"]
    f_sic = out["sic_phase"]
    f_si3n4 = out["si3n4_phase"]
    phase_sum = (f_c + f_sio2 + f_sic + f_si3n4).replace(0, np.nan)

    f_c_norm = f_c / phase_sum
    f_sio2_norm = f_sio2 / phase_sum
    f_sic_norm = f_sic / phase_sum
    f_si3n4_norm = f_si3n4 / phase_sum

    out["conductive_network_index"] = (
        1.00 * f_c_norm + 0.15 * f_sic_norm - 0.80 * f_sio2_norm - 0.60 * f_si3n4_norm
    )
    ceramic = f_sio2_norm + f_sic_norm + f_si3n4_norm
    out["ceramic_confinement_index"] = ceramic / (f_c_norm + 1e-9)
    out["effective_bandgap_proxy_ev"] = (
        0.0 * f_c_norm + 2.3 * f_sic_norm + 9.0 * f_sio2_norm + 5.0 * f_si3n4_norm
    )

    chi_free_c = 2.55
    chi_sio2 = (1.90 + 2 * 3.44) / 3
    chi_sic = (1.90 + 2.55) / 2
    chi_si3n4 = (3 * 1.90 + 4 * 3.04) / 7
    out["phase_weighted_electronegativity"] = (
        chi_free_c * f_c_norm + chi_sio2 * f_sio2_norm + chi_sic * f_sic_norm + chi_si3n4 * f_si3n4_norm
    )
    out["phase_separation_index"] = f_c_norm + f_sic_norm
    return out


def add_process_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    numeric = [
        "pyrolysis_temp_c", "pyrolysis_time_h", "crosslink_temp_c", "crosslink_time_h",
        "first_cycling_current_ma_g", "cycling_current_ma_g", "surface_area_m2_g",
    ]
    out = _to_numeric(out, numeric)

    if "crosslinking_method" not in out.columns:
        out["crosslinking_method"] = "unknown"
    if "pre_pyrolysis_method" not in out.columns:
        out["pre_pyrolysis_method"] = out["crosslinking_method"]

    method_source = out["pre_pyrolysis_method"].fillna(out["crosslinking_method"]).fillna("unknown")
    method = method_source.astype(str).str.lower().str.strip()
    out["crosslinking_used"] = (~method.isin(["none", "no", "-", "unknown", "nan", ""])).astype(int)

    out["pre_pyrolysis_method"] = method_source.replace({"nan": "none"}).fillna("none")
    out["pre_pyrolysis_temp_c"] = out.get("crosslink_temp_c", np.nan)
    out["pre_pyrolysis_time_h"] = out.get("crosslink_time_h", np.nan)
    out["pre_pyrolysis_atmosphere"] = out.get("crosslink_atmosphere", "unknown")

    method_clean = method.str.replace("–", "-", regex=False).str.replace("_", "-", regex=False)
    out["pre_pyrolysis_method_group"] = np.select(
        [
            method_clean.isin(["none", "no", "-", "unknown", "nan", ""]),
            method_clean.str.contains("sol-blending|solution-blending|solvent-blending", regex=True, na=False),
            method_clean.str.contains("sol-gel.*thermal|thermal.*sol-gel", regex=True, na=False),
            method_clean.str.contains("sol-gel|sol-ger", regex=True, na=False),
            method_clean.str.contains(r"\buv\b|photo", regex=True, na=False),
            method_clean.str.contains("peroxide|dicumyl|pt|catalyst|catalyzed|catalysed", regex=True, na=False),
            method_clean.str.contains("staged-pyrolysis|thermal-pre-treatment|pyrolysis", regex=True, na=False),
            method_clean.str.contains("thermal", regex=True, na=False),
            method_clean.str.contains("autoclave|hydrothermal", regex=True, na=False),
        ],
        [
            "none",
            "sol_blending",
            "sol_gel_thermal",
            "sol_gel",
            "uv_crosslinking",
            "catalyzed_crosslinking",
            "staged_pyrolysis",
            "thermal_crosslinking",
            "solvothermal_autoclave",
        ],
        default="other_pre_pyrolysis",
    )
    out["pre_pyrolysis_used"] = (out["pre_pyrolysis_method_group"] != "none").astype(int)

    if "post_pyrolysis" not in out.columns:
        out["post_pyrolysis"] = "none"
    post = out["post_pyrolysis"].fillna("none").astype(str).str.lower().str.strip().str.replace(" ", "_")
    out["post_pyrolysis_group"] = np.select(
        [
            post.isin(["none", "no", "-", "unknown", "nan", ""]),
            post.str.contains("hf", regex=True, na=False),
            post.str.contains("naoh", regex=True, na=False),
            post.str.contains("koh", regex=True, na=False),
            post.str.contains("etch", regex=True, na=False),
        ],
        ["none", "hf_etching", "naoh_etching", "koh_etching", "other_etching"],
        default="other_post_pyrolysis",
    )
    out["post_pyrolysis_used"] = (out["post_pyrolysis_group"] != "none").astype(int)

    out["pyrolysis_thermal_budget"] = out.get("pyrolysis_temp_c", np.nan) * out.get("pyrolysis_time_h", np.nan)
    out["crosslinking_thermal_budget"] = out.get("crosslink_temp_c", np.nan) * out.get("crosslink_time_h", np.nan)
    out["pre_pyrolysis_thermal_budget"] = out.get("pre_pyrolysis_temp_c", np.nan) * out.get("pre_pyrolysis_time_h", np.nan)
    out["log_surface_area_m2_g"] = np.log1p(out.get("surface_area_m2_g", np.nan))

    if "pyrolysis_atmosphere" not in out.columns:
        out["pyrolysis_atmosphere"] = "unknown"
    atm = out["pyrolysis_atmosphere"].fillna("unknown").astype(str).str.lower().str.replace(" ", "")
    out["pyrolysis_atmosphere_group"] = np.select(
        [
            atm.str.contains("co2|o2|oxygen", regex=True, na=False),
            atm.str.fullmatch("h2|hydrogen", na=False),
            atm.str.contains("ar|argon|n2|nitrogen|inert|forminggas|h2|air|unknown|nan", regex=True, na=False),
        ],
        ["oxidizing", "reducing", "inert"],
        default="inert",
    )
    return out


def add_voltage_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    defaults = {"voltage_min_v": 0.01, "voltage_max_v": 3.0, "voltage_window_v": 2.99}
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    out = _to_numeric(out, list(defaults))
    for col, default in defaults.items():
        out[col] = out[col].fillna(default)
    missing_window = out["voltage_window_v"].isna() & out["voltage_min_v"].notna() & out["voltage_max_v"].notna()
    out.loc[missing_window, "voltage_window_v"] = out.loc[missing_window, "voltage_max_v"] - out.loc[missing_window, "voltage_min_v"]
    return out


def add_electrolyte_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    if "electrolyte" not in out.columns:
        out["electrolyte"] = "unknown"
    elec = out["electrolyte"].fillna("unknown").astype(str).str.lower()
    species = {
        "ec": r"(?<![a-z])ec(?![a-z])|ethylene\s+carbonate",
        "dec": r"(?<![a-z])dec(?![a-z])|diethyl\s+carbonate",
        "dmc": r"(?<![a-z])dmc(?![a-z])|dimethyl\s+carbonate",
        "emc": r"(?<![a-z])emc(?![a-z])|ethyl\s+methyl\s+carbonate",
        "pc": r"(?<![a-z])pc(?![a-z])|propylene\s+carbonate",
        "fec": r"(?<![a-z])fec(?![a-z])|fluoroethylene\s+carbonate",
        "vc": r"(?<![a-z])vc(?![a-z])|vinylene\s+carbonate",
        "diglyme": r"diglyme",
        "dme": r"(?<![a-z])dme(?![a-z])|dimethoxyethane",
    }
    for name, pattern in species.items():
        out[f"{name}_present"] = elec.str.contains(pattern, regex=True, na=False).astype(int)
    solvent_cols = [f"{n}_present" for n in ["ec", "dec", "dmc", "emc", "pc", "diglyme", "dme"]]
    additive_cols = ["fec_present", "vc_present"]
    out["solvent_count"] = out[solvent_cols].sum(axis=1)
    out["additive_count"] = out[additive_cols].sum(axis=1)
    out["electrolyte_known"] = (~elec.isin(["unknown", "nan", "", "-"])).astype(int)
    return out


def add_precursor_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    if "polymer" not in out.columns:
        out["polymer"] = "unknown"
    p = out["polymer"].fillna("unknown").astype(str).str.lower()
    p_clean = p.str.replace("％", "%", regex=False)
    dvb_pattern = r"\bdvb\b|divinylbenzene|divinylbenzen"
    p_base = (
        p_clean
        .str.replace(r"\s*\+\s*(?:divinylbenzene|divinylbenzen)?\s*\(?\s*dvb\s*\)?\s*(?:\([^)]*\))?", "", regex=True)
        .str.replace(r"\s*\+\s*(?:divinylbenzene|divinylbenzen)\s*\(?\s*dvb?\s*\)?\s*(?:\([^)]*\))?", "", regex=True)
        .str.replace(r"\s+with\s+(?:\d+(?:\.\d+)?\s*wt\.?\s*%\s*)?(?:dvb|divinylbenzene|divinylbenzen).*", "", regex=True)
        .str.strip()
    )
    has_phtes = p_base.str.contains("phtes|phenyltriethoxysilane|phtms|phenyltrimethoxysilane|triethoxyphenylsilane|phenyltrimethoxysilane", regex=True, na=False)
    has_vtes = p_base.str.contains("vtes|vinyltriethoxysilane|vtms|vinyltrimethoxysilane", regex=True, na=False)
    has_mtes = p_base.str.contains("mtes|methyltriethoxysilane|mtms|methyltrimethoxysilane|methoxy silane", regex=True, na=False)
    has_teos = p_base.str.contains("teos|tetraethoxysilane", regex=True, na=False)
    has_prtes = p_base.str.contains("prtes|triethoxypropylsilane", regex=True, na=False)
    has_phenyl_polysiloxane = p_base.str.contains("phenyl|pmps|rd-?684|spr-?684|rd-?688|mt resin|mq resin|methylvinylphenyl|methylphenyl|diphenyl|polyphenyl", regex=True, na=False)
    has_alkyl_polysiloxane = p_base.str.contains("methyl|alkyl|dimethyl|trimethyl|pmhs|phms|pdms|rd-?212|polyhydromethyl|polyhydridomethyl|poly\\(methylhydrogen", regex=True, na=False)
    has_vinyl_polysiloxane = p_base.str.contains("vinyl|rd-?212|tmtvs|ttcs|cyclotetrasiloxane|tpts|dtds|ddts", regex=True, na=False)
    is_polysiloxane_like = p_base.str.contains("rd-?684|spr-?684|rd-?688|rd-?212|pmhs|phms|pdms|polyhydro|polyhydrido|polysiloxane|siloxane|tmtvs|ttcs|cyclotetrasiloxane|mt resin|mq resin|mq resins|methylvinylphenyl.*mt|pmps", regex=True, na=False)

    conditions = [
        p_base.str.contains("rd-?684|spr-?684|polyramic", regex=True, na=False),
        p_base.str.contains("rd-?688", regex=True, na=False),
        p_base.str.contains("rd-?212", regex=True, na=False),
        p_base.str.contains("pmhs|phms|polyhydro|polyhydrido", regex=True, na=False),
        p_base.str.contains(r"pms\s*mk|polysilsesquioxane", regex=True, na=False),
        has_teos & has_phtes,
        has_teos & has_prtes,
        has_phtes & has_mtes,
        has_phtes & has_vtes,
        has_vtes & has_mtes,
        has_phtes,
        has_vtes,
        has_mtes,
        has_teos,
        has_prtes,
        p_base.str.contains("ttcs|tmtvs|cyclotetrasiloxane", regex=True, na=False),
        p_base.str.contains("silicone oil|kf-", regex=True, na=False),
        p_base.str.contains("sol-gel|xing1997", regex=True, na=False),
    ]
    choices = [
        "rd_684_resin",
        "rd_688_resin",
        "rd_212_resin",
        "pmhs_phms",
        "pms_mk",
        "phtes_teos_derived_polysiloxane",
        "prtes_teos_derived_polysiloxane",
        "phtes_mtes_derived_polysiloxane",
        "phtes_vtes_derived_polysiloxane",
        "vtes_mtes_derived_polysiloxane",
        "phtes_derived_polysiloxane",
        "vtes_derived_polysiloxane",
        "mtes_derived_polysiloxane",
        "teos_derived_silica_polysiloxane",
        "prtes_derived_polysiloxane",
        "cyclic_vinyl_siloxane",
        "silicone_oil",
        "xing_sol_gel_legacy",
    ]
    out["precursor_family"] = np.select(conditions, choices, default="other")
    broad_conditions = [
        p_base.str.contains("polysilazane|polyorganosilazane|polyureasilazane|silazane|durazane|htt1800|vl20|ceraset|silsesquiazane", regex=True, na=False),
        p_base.str.contains("carbodiimide|silylcarbodiimide", regex=True, na=False),
        p_base.str.contains("polycarbosilane|carbosilane|pcs\\b", regex=True, na=False),
        p_base.str.contains(r"pms\s*mk|polysilsesquioxane|silsesquioxane", regex=True, na=False),
        has_teos & has_phtes,
        has_teos & has_prtes,
        has_phtes & has_mtes,
        has_phtes & has_vtes,
        has_vtes & has_mtes,
        has_phtes,
        has_vtes,
        has_mtes | p_base.str.contains("epoxy-silane|epoxy silane|alkoxysilane", regex=True, na=False),
        has_teos,
        is_polysiloxane_like & has_phenyl_polysiloxane & has_alkyl_polysiloxane,
        is_polysiloxane_like & has_phenyl_polysiloxane,
        is_polysiloxane_like & has_vinyl_polysiloxane,
        is_polysiloxane_like & has_alkyl_polysiloxane,
        is_polysiloxane_like,
        p_base.str.contains("silicone oil|kf-", regex=True, na=False),
        p_base.str.contains("pitch|graphene|carbon|hard carbons|starch|sucrose|phenolic|phenohc|acenaphthylene|polystyrene", regex=True, na=False),
        p_base.str.contains("polysilane", regex=True, na=False),
        p_base.str.contains("sol-gel|xing1997", regex=True, na=False),
    ]
    broad_choices = [
        "polysilazane",
        "silylcarbodiimide",
        "polycarbosilane",
        "polysilsesquioxane",
        "phtes_teos_derived_polysiloxane",
        "prtes_teos_derived_polysiloxane",
        "phtes_mtes_derived_polysiloxane",
        "phtes_vtes_derived_polysiloxane",
        "vtes_mtes_derived_polysiloxane",
        "phtes_derived_polysiloxane",
        "vtes_derived_polysiloxane",
        "mtes_derived_polysiloxane",
        "teos_derived_silica_polysiloxane",
        "phenylalkyl_polysiloxane",
        "phenyl_polysiloxane",
        "vinyl_polysiloxane",
        "alkyl_polysiloxane",
        "polysiloxane",
        "silicone_oil",
        "carbon_rich_blend",
        "organopolysilane_copolymer",
        "sol_gel_legacy",
    ]
    out["polymer_family_broad"] = np.select(broad_conditions, broad_choices, default="other")
    out["hf_treated"] = p.str.contains(r"\bhf\b", regex=True, na=False).astype(int)
    out["dvb_modification"] = p_clean.str.contains(dvb_pattern, regex=True, na=False).astype(int)
    out["dvb_present"] = out["dvb_modification"]
    out["dvb_ratio_to_base"] = np.nan
    out["dvb_wt_pct_nominal"] = np.nan

    uniform_ratio = p_clean.str.extract(
        r"(?:dvb|divinylbenzene|divinylbenzen)[^\n]{0,30}\((\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*wt",
        expand=True,
    )
    ratio_after = p_clean.str.extract(
        r"(?:dvb|divinylbenzene|divinylbenzen)[^\d]{0,20}(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)",
        expand=True,
    )
    ratio_before = p_clean.str.extract(
        r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)[^\n]{0,40}(?:dvb|divinylbenzene|divinylbenzen)",
        expand=True,
    )
    wt_pct = p_clean.str.extract(
        r"(\d+(?:\.\d+)?)\s*wt\.?\s*%[^\n]{0,20}(?:dvb|divinylbenzene|divinylbenzen)|"
        r"(?:dvb|divinylbenzene|divinylbenzen)[^\d]{0,20}(\d+(?:\.\d+)?)\s*wt\.?\s*%",
        expand=True,
    )
    ternary_after = p_clean.str.extract(
        r"(?:dvb|divinylbenzene|divinylbenzen)[^\n]{0,40}\((\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*wt",
        expand=True,
    )
    ternary_before = p_clean.str.extract(
        r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)[^\n]{0,60}(?:dvb|divinylbenzene|divinylbenzen)",
        expand=True,
    )

    ternary_after_valid = ternary_after.notna().all(axis=1)
    if ternary_after_valid.any():
        a = pd.to_numeric(ternary_after.loc[ternary_after_valid, 0], errors="coerce")
        b = pd.to_numeric(ternary_after.loc[ternary_after_valid, 1], errors="coerce")
        y = pd.to_numeric(ternary_after.loc[ternary_after_valid, 2], errors="coerce")
        out.loc[ternary_after_valid, "dvb_ratio_to_base"] = y / (a + b)
        out.loc[ternary_after_valid, "dvb_wt_pct_nominal"] = 100.0 * y / (a + b + y)

    ternary_before_valid = ternary_before.notna().all(axis=1) & out["dvb_ratio_to_base"].isna()
    if ternary_before_valid.any():
        a = pd.to_numeric(ternary_before.loc[ternary_before_valid, 0], errors="coerce")
        b = pd.to_numeric(ternary_before.loc[ternary_before_valid, 1], errors="coerce")
        y = pd.to_numeric(ternary_before.loc[ternary_before_valid, 2], errors="coerce")
        out.loc[ternary_before_valid, "dvb_ratio_to_base"] = y / (a + b)
        out.loc[ternary_before_valid, "dvb_wt_pct_nominal"] = 100.0 * y / (a + b + y)

    uniform_valid = uniform_ratio.notna().all(axis=1) & out["dvb_ratio_to_base"].isna()
    out.loc[uniform_valid, "dvb_ratio_to_base"] = (
        pd.to_numeric(uniform_ratio.loc[uniform_valid, 1], errors="coerce")
        / pd.to_numeric(uniform_ratio.loc[uniform_valid, 0], errors="coerce")
    )

    after_valid = ratio_after.notna().all(axis=1) & out["dvb_ratio_to_base"].isna()
    out.loc[after_valid, "dvb_ratio_to_base"] = (
        pd.to_numeric(ratio_after.loc[after_valid, 1], errors="coerce")
        / pd.to_numeric(ratio_after.loc[after_valid, 0], errors="coerce")
    )

    before_valid = ratio_before.notna().all(axis=1) & out["dvb_ratio_to_base"].isna()
    out.loc[before_valid, "dvb_ratio_to_base"] = (
        pd.to_numeric(ratio_before.loc[before_valid, 1], errors="coerce")
        / pd.to_numeric(ratio_before.loc[before_valid, 0], errors="coerce")
    )

    nominal_pct = pd.to_numeric(wt_pct.iloc[:, 0], errors="coerce").combine_first(
        pd.to_numeric(wt_pct.iloc[:, 1], errors="coerce")
    )
    out["dvb_wt_pct_nominal"] = out["dvb_wt_pct_nominal"].combine_first(nominal_pct)
    missing_ratio = out["dvb_ratio_to_base"].isna() & out["dvb_wt_pct_nominal"].notna()
    out.loc[missing_ratio, "dvb_ratio_to_base"] = out.loc[missing_ratio, "dvb_wt_pct_nominal"] / 100.0
    parsed_ratio = out["dvb_ratio_to_base"].notna() & out["dvb_wt_pct_nominal"].isna()
    out.loc[parsed_ratio, "dvb_wt_pct_nominal"] = (
        100.0
        * out.loc[parsed_ratio, "dvb_ratio_to_base"]
        / (1.0 + out.loc[parsed_ratio, "dvb_ratio_to_base"])
    )
    no_dvb = out["dvb_modification"].eq(0)
    out.loc[no_dvb, "dvb_ratio_to_base"] = 0.0
    out.loc[no_dvb, "dvb_wt_pct_nominal"] = 0.0
    out["dvb_ratio_to_base"] = out["dvb_ratio_to_base"].clip(lower=0.0, upper=5.0)
    out["dvb_wt_pct_nominal"] = out["dvb_wt_pct_nominal"].clip(lower=0.0, upper=100.0)

    out["phenyl_present"] = p.str.contains("phenyl|phtes|phtms|ph", regex=True, na=False).astype(int)
    out["vinyl_present"] = p_base.str.contains("vinyl|vtes", regex=True, na=False).astype(int)
    return out


def add_electrochemical_performance_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    out = _to_numeric(out, [
        "irreversible_capacity_mah_g", "coulombic_efficiency_pct",
        "cycling_reversible_capacity_mah_g", "capacity_retention_pct",
        "cycling_numbers", "first_cycling_current_ma_g", "cycling_current_ma_g",
    ])
    # If Coulombic efficiency is absent but first reversible/irreversible capacities are present,
    # estimate first-cycle CE = reversible / (reversible + irreversible) * 100.
    if "coulombic_efficiency_pct" not in out.columns:
        out["coulombic_efficiency_pct"] = np.nan
    if "irreversible_capacity_mah_g" in out.columns and TARGET in out.columns:
        denom = out[TARGET] + out["irreversible_capacity_mah_g"]
        estimate = np.where(denom > 0, out[TARGET] / denom * 100, np.nan)
        out["coulombic_efficiency_pct"] = out["coulombic_efficiency_pct"].fillna(pd.Series(estimate, index=out.index))
    return out


def prepare_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(dataframe)
    out = add_composition_features(out)
    out = add_phase_physics_features(out)
    out = add_process_features(out)
    out = add_voltage_features(out)
    out = add_electrolyte_features(out)
    out = add_precursor_features(out)
    out = add_electrochemical_performance_features(out)
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out


def get_model_matrix(dataframe: pd.DataFrame):
    """Return X, y, and engineered dataframe using the expanded deployment schema."""
    df = prepare_features(dataframe)
    df_ml = df.dropna(subset=[TARGET]).copy()
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in df_ml.columns:
            df_ml[col] = np.nan
    X = df_ml[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df_ml[TARGET]
    return X, y, df_ml
