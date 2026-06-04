"""Virtual SiOC/SiOCN Anode Designer Streamlit app.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from src.features import DISPLAY_FEATURE_GROUPS, TARGET, prepare_features, read_sioc_csv
except ModuleNotFoundError:
    from features import DISPLAY_FEATURE_GROUPS, TARGET, prepare_features, read_sioc_csv


DATA_PATH = Path("data/sioc_battery_capacity_clean_updated.csv")
DISCOVERY_MODEL_PATH = Path("models/sioc_final_discovery_model.joblib")
STABLE_MODEL_PATH = Path("models/sioc_stable_capacity_model.joblib")
STABLE_DIAGNOSTIC_MODEL_PATH = Path("models/sioc_stable_capacity_diagnostic_model.joblib")
IRREVERSIBLE_MODEL_PATH = Path("models/sioc_irreversible_capacity_model.joblib")
CE_MODEL_PATH = Path("models/sioc_coulombic_efficiency_model.joblib")
CE_DIAGNOSTIC_MODEL_PATH = Path("models/sioc_coulombic_efficiency_diagnostic_model.joblib")
APP_TARGET_MODELS_PATH = Path("models/sioc_app_target_models.joblib")
PUBLIC_REFERENCE_KEY = "_public_reference_raw"
STABLE_TARGET = "cycling_reversible_capacity_mah_g"
IRREVERSIBLE_TARGET = "irreversible_capacity_mah_g"
CE_TARGET = "coulombic_efficiency_pct"
LOW_RATE_FIRST_CURRENT_MA_G = 18.6
LOW_RATE_CYCLING_CURRENT_MA_G = 37.2
GRAPHITE_CAPACITY_MAH_G = 372.0

TARGET_METADATA = {
    TARGET: {
        "label": "First-cycle reversible capacity",
        "unit": "mAh/g",
        "context": "Higher is generally better, interpreted at low-rate 0-3 V literature conditions.",
        "compare_graphite": True,
    },
    IRREVERSIBLE_TARGET: {
        "label": "First-cycle irreversible capacity",
        "unit": "mAh/g",
        "context": "Lower is generally better because it indicates less first-cycle lithium loss.",
        "compare_graphite": False,
    },
    CE_TARGET: {
        "label": "First-cycle Coulombic efficiency",
        "unit": "%",
        "context": "Higher is generally better. This model is moderately predictive but still literature-limited.",
        "compare_graphite": False,
    },
    STABLE_TARGET: {
        "label": "Cycled reversible capacity",
        "unit": "mAh/g",
        "context": "Capacity at the selected cycle number, interpreted at low-rate 0-3 V literature conditions.",
        "compare_graphite": True,
    },
}

METHOD_LABELS = {
    "none": "None",
    "sol_blending": "Sol blending",
    "sol_gel": "Sol-gel",
    "sol_gel_thermal": "Sol-gel + thermal",
    "uv_crosslinking": "UV crosslinking",
    "thermal_crosslinking": "Thermal crosslinking",
    "catalyzed_crosslinking": "Catalyzed crosslinking",
    "staged_pyrolysis": "Staged pre-pyrolysis",
    "solvothermal_autoclave": "Solvothermal/autoclave",
}

METHOD_TIPS = {
    "none": "No deliberate pre-pyrolysis step. Temperature/time are set to 0 for prediction.",
    "sol_blending": "Literature examples are narrow: about 100 C and 1 h.",
    "sol_gel": "Sol-gel entries are typically low-temperature aging/gelation, not high-temperature crosslinking.",
    "sol_gel_thermal": "Use for sol-gel followed by a defined thermal treatment.",
    "uv_crosslinking": "UV crosslinking in this dataset is near room temperature. 500 C would be thermal treatment, not UV.",
    "thermal_crosslinking": "Thermal pre-treatment/crosslinking before pyrolysis.",
    "catalyzed_crosslinking": "Catalyst- or initiator-assisted crosslinking, often peroxide/Pt/AIBN-related.",
    "staged_pyrolysis": "A low-temperature ceramic conversion step before the final pyrolysis.",
    "solvothermal_autoclave": "Sparse class. Treat predictions as exploratory.",
}

POLYMER_FAMILY_TEMPLATES = {
    "Phenylalkyl polysiloxane": "methylvinylphenyl polysiloxane resin",
    "Phenylalkyl polysiloxane + pyrrole/PVP": "methylphenylvinylhydrogen polysiloxane SILRES H62C + pyrrole + graphene oxide",
    "Alkyl polysiloxane": "polyhydromethylsiloxane PMHS PHMS",
    "Vinyl polysiloxane": "tetramethyltetravinylcyclotetrasiloxane TMTVS vinyl methyl polysiloxane resin RD-212",
    "Vinyl polysiloxane + PVP": "1,3-divinyltetramethyldisiloxane DTDS + polyvinylpyrrolidone PVP",
    "Phenyl polysiloxane": "polyphenylsesquisiloxane PPSSO phenyl silicone oil KF-54",
    "Polysiloxane": "generic siloxane resin",
    "Polysilazane / polyorganosilazane": "polyorganosilazane HTT1800 polysilazane VL20 Durazane",
    "Polycarbosilane": "polycarbosilane PCS",
    "Organopolysilane copolymer": "Polysilane copolymer poly(dimethylsilylene-co-methylphenylsilylene) PSS-120 (Me2Si)0.8(PhMeSi)0.2",
    "Polysilsesquioxane": "polysilsesquioxane PMS MK",
    "Silylcarbodiimide / silylsesquiazane": "polyphenylvinylsilylcarbodiimide bis(trimethylsilyl)carbodiimide",
    "PhTES-derived polysiloxane": "polysiloxanes from phenyltriethoxysilane PhTES",
    "PhTES + TEOS derived polysiloxane": "polysiloxanes from phenyltriethoxysilane PhTES + tetraethoxysilane TEOS",
    "PhTES + MTES derived polysiloxane": "polysiloxanes from phenyltriethoxysilane PhTES + methyltriethoxysilane MTES",
    "PhTES + VTES derived polysiloxane": "polysiloxanes from phenyltriethoxysilane PhTES + vinyltriethoxysilane VTES",
    "VTES + MTES derived polysiloxane": "polysiloxanes from vinyltriethoxysilane VTES + methyltriethoxysilane MTES",
    "PrTES + TEOS derived polysiloxane": "polysiloxanes from tetraethoxysilane TEOS + triethoxypropylsilane PrTES",
    "MTES-derived polysiloxane": "polysiloxanes from methyltriethoxysilane MTES",
    "VTES-derived polysiloxane": "polysiloxanes from vinyltriethoxysilane VTES",
    "TEOS-derived silica/polysiloxane": "polysiloxanes from tetraethoxysilane TEOS",
    "Alkoxysilane-derived polysiloxane": "phenyltriethoxysilane PhTES vinyltriethoxysilane VTES",
    "Silicone oil": "silicone oil KF-96",
    "Carbon-rich blend": "pitch polysilane blend",
    "Other / custom": "custom precursor",
}

POLYMER_COMPATIBILITY = {
    "Phenylalkyl polysiloxane": "SiOC; phenyl plus alkyl/methyl siloxane resins. This is the dominant polysiloxane class in the dataset.",
    "Phenylalkyl polysiloxane + pyrrole/PVP": "Low-N SiOC route where N comes from pyrrole/PVP additive, not from the polysiloxane backbone.",
    "Alkyl polysiloxane": "SiOC; methyl/hydrido/alkyl siloxanes such as PMHS/PHMS.",
    "Vinyl polysiloxane": "SiOC; vinyl-rich siloxanes and cyclic vinyl siloxanes.",
    "Vinyl polysiloxane + PVP": "Low-N SiOC route where N comes from polyvinylpyrrolidone additive, not from the vinyl siloxane.",
    "Phenyl polysiloxane": "SiOC; phenyl-rich siloxanes. Sparse class, so treat predictions cautiously.",
    "Polysiloxane": "Generic siloxane resin fallback when the side-group chemistry is unclear.",
    "Polysilazane / polyorganosilazane": "Best chemical choice for N-containing SiOCN/SiCN-like compositions.",
    "Polycarbosilane": "SiC-rich ceramic polymer family; useful for low-O/high-SiC targets.",
    "Organopolysilane copolymer": "Methyl/phenyl substituted Si-Si copolymer route from the Xing dataset; carbon comes from the organic substituents or blend partner.",
    "Polysilsesquioxane": "O-rich SiOC; good for high Si/O targets.",
    "Silylcarbodiimide / silylsesquiazane": "N-rich SiCN/SiOCN-like chemistry.",
    "PhTES-derived polysiloxane": "Phenyl alkoxysilane sol-gel route; phenyl groups provide carbon after pyrolysis.",
    "PhTES + TEOS derived polysiloxane": "Mixed phenyl/TEOS sol-gel route; balances phenyl-derived carbon with silica-rich TEOS.",
    "PhTES + MTES derived polysiloxane": "Mixed phenyl/methyl sol-gel route; carbon comes from phenyl and methyl substituents.",
    "PhTES + VTES derived polysiloxane": "Mixed phenyl/vinyl sol-gel route; carbon comes from phenyl and reactive vinyl substituents.",
    "VTES + MTES derived polysiloxane": "Mixed vinyl/methyl sol-gel route; carbon comes from vinyl and methyl substituents.",
    "PrTES + TEOS derived polysiloxane": "Mixed propyl/TEOS sol-gel route; propyl groups provide carbon in a silica-rich network.",
    "MTES-derived polysiloxane": "Methyl alkoxysilane sol-gel route; O-rich with modest methyl-derived carbon.",
    "VTES-derived polysiloxane": "Vinyl alkoxysilane sol-gel route; O-rich with reactive vinyl carbon source.",
    "TEOS-derived silica/polysiloxane": "Very O-rich silica-like sol-gel network; carbon mainly requires an organic co-precursor.",
    "Alkoxysilane-derived polysiloxane": "Generic O-rich alkoxysilane-derived polysiloxane route; no meaningful N source.",
    "Silicone oil": "C-rich SiOC; no meaningful N source.",
    "Carbon-rich blend": "Very carbon-rich blends; exploratory in this model.",
    "Other / custom": "Use only when the precursor is outside the recognized literature families.",
}


st.set_page_config(
    page_title="SiOC/SiOCN Anode Designer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div {padding-top: 1rem;}
    .hero {
        padding: 1.2rem 1.35rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #0f2533 0%, #174f59 52%, #236f68 100%);
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.16);
    }
    .hero-grid {display: flex; align-items: center; justify-content: space-between; gap: 1rem;}
    .hero-title {min-width: 0;}
    .hero h1 {font-size: 2rem; margin: 0 0 0.2rem 0;}
    .hero p {font-size: 1rem; opacity: 0.93; margin: 0;}
    .energy-mark {
        width: 92px; min-width: 92px; height: 54px; border: 3px solid rgba(255,255,255,0.9);
        border-radius: 10px; position: relative; background: rgba(255,255,255,0.08);
        box-shadow: inset 0 0 18px rgba(88, 214, 141, 0.22), 0 8px 24px rgba(0,0,0,0.18);
    }
    .energy-mark:after {
        content: ""; position: absolute; right: -10px; top: 16px; width: 8px; height: 20px;
        border-radius: 0 4px 4px 0; background: rgba(255,255,255,0.9);
    }
    .energy-fill {
        position: absolute; left: 6px; top: 6px; bottom: 6px; width: 62px; border-radius: 6px;
        background: linear-gradient(90deg, #62d26f, #d3f75d);
    }
    .energy-bolt {
        position: absolute; left: 50%; top: 50%; transform: translate(-50%, -52%);
        color: #0f2533; font-size: 30px; font-weight: 900;
        text-shadow: 0 1px 0 rgba(255,255,255,0.35);
    }
    .note-card {
        padding: 0.9rem 1rem;
        border-radius: 8px;
        background: #f7fafc;
        border: 1px solid #e6edf3;
        height: 100%;
    }
    .small-muted {font-size: 0.86rem; color: #607080;}
    .range-ok {color: #146c43; font-weight: 600;}
    .range-warn {color: #9a3412; font-weight: 600;}
    .formula-pill {
        display: inline-block; padding: 0.35rem 0.55rem; margin: 0.15rem 0.15rem 0.15rem 0;
        border-radius: 999px; background: #eef7f4; border: 1px solid #cae7de; font-weight: 650;
    }
    .stTabs [role="tablist"] {gap: 0.35rem;}
    .stTabs [role="tab"] {
        border: 1px solid #b8d8d0; border-radius: 8px; padding: 0.45rem 0.75rem;
        background: #edf8f5; color: #12323c; font-weight: 800;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #174f59 0%, #236f68 100%) !important;
        color: white !important; border-color: #174f59;
    }
    div[role="radiogroup"] label {
        border: 1px solid #c7dce6; border-radius: 8px; padding: 0.35rem 0.55rem;
        background: #f5fbfd; font-weight: 750; margin-right: 0.25rem; color: #12323c;
    }
    div[role="radiogroup"] label:has(input:checked) {
        background: #174f59 !important; color: #ffffff !important; border-color: #174f59;
    }
    div[role="radiogroup"] label:has(input:checked) *,
    div[role="radiogroup"] label:has(input:checked) p,
    div[role="radiogroup"] label:has(input:checked) span {
        color: #ffffff !important;
    }
    .boxed-label {
        display: inline-block; padding: 0.35rem 0.6rem; border-radius: 8px;
        background: #fff4e6; color: #7a3e00; border: 1px solid #ffd59a;
        font-weight: 850; margin: 0.25rem 0 0.45rem 0;
    }
    .prediction-card {
        min-height: 118px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.85rem;
        border-radius: 8px;
        border: 1px solid #d8e6ec;
        background: linear-gradient(180deg, #ffffff 0%, #f7fbfc 100%);
        box-shadow: 0 8px 20px rgba(15, 37, 51, 0.06);
        border-top: 5px solid var(--accent);
    }
    .prediction-label {
        color: #526575;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        text-transform: none;
        margin-bottom: 0.25rem;
    }
    .prediction-value {
        color: #102334;
        font-size: 2.0rem;
        font-weight: 850;
        line-height: 1.05;
        white-space: nowrap;
    }
    .prediction-unit {
        color: #536879;
        font-size: 1.02rem;
        font-weight: 700;
        margin-left: 0.2rem;
    }
    .prediction-note {
        color: #7a8791;
        font-size: 0.82rem;
        margin-top: 0.55rem;
        line-height: 1.25;
    }
    .prediction-card.soft {
        min-height: 105px;
        background: #fbfdfe;
    }
    .app-footer {
        margin-top: 1.4rem;
        padding: 0.85rem 1rem;
        border-top: 1px solid #dbe7ec;
        color: #526575;
        font-size: 0.9rem;
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .footer-links {
        display: flex;
        gap: 0.55rem;
        align-items: center;
    }
    .footer-link {
        color: #174f59 !important;
        text-decoration: none !important;
        font-weight: 800;
        border: 1px solid #bfd8d3;
        border-radius: 999px;
        padding: 0.22rem 0.55rem;
        background: #f3faf8;
        display: inline-flex;
        gap: 0.3rem;
        align-items: center;
    }
    .footer-link:hover {
        background: #e4f3ef;
        border-color: #8dbfb5;
    }
    .footer-icon {
        width: 15px;
        height: 15px;
        display: inline-block;
        vertical-align: -2px;
    }
    @media (max-width: 900px) {
        .hero-grid {align-items: flex-start;}
        .hero h1 {font-size: 1.6rem;}
        .prediction-value {font-size: 1.7rem;}
        .prediction-unit {font-size: 0.92rem;}
        .app-footer {align-items: flex-start;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle(path: Path) -> dict | None:
    if not path.exists():
        return None
    return joblib.load(path)


def prediction_card(
    label: str,
    value: str,
    unit: str = "",
    note: str = "",
    accent: str = "#236f68",
    soft: bool = False,
) -> None:
    unit_html = f'<span class="prediction-unit">{escape(unit)}</span>' if unit else ""
    note_html = f'<div class="prediction-note">{escape(note)}</div>' if note else ""
    soft_class = " soft" if soft else ""
    st.markdown(
        f"""
        <div class="prediction-card{soft_class}" style="--accent: {accent};">
            <div class="prediction-label">{label}</div>
            <div class="prediction-value">{escape(value)}{unit_html}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_reference_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_sioc_csv(str(path))


@st.cache_data
def load_engineered_reference(path: Path) -> pd.DataFrame:
    raw = load_reference_data(path)
    if raw.empty:
        return pd.DataFrame()
    return prepare_features(raw)


def public_reference_from_bundle(bundle: dict | None) -> pd.DataFrame:
    if not isinstance(bundle, dict):
        return pd.DataFrame()
    public_ref = bundle.get(PUBLIC_REFERENCE_KEY)
    if isinstance(public_ref, pd.DataFrame):
        return public_ref.copy()
    return pd.DataFrame()


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return ""
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    if text.startswith("doi.org/"):
        text = text.replace("doi.org/", "", 1)
    return f"https://doi.org/{text}"


def get_numeric_range(df: pd.DataFrame, col: str, fallback: tuple[float, float, float]) -> dict:
    if col not in df.columns:
        mn, med, mx = fallback
        return {"min": mn, "q05": mn, "median": med, "q95": mx, "max": mx}
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        mn, med, mx = fallback
        return {"min": mn, "q05": mn, "median": med, "q95": mx, "max": mx}
    return {
        "min": float(s.min()),
        "q05": float(s.quantile(0.05)),
        "median": float(s.median()),
        "q95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def method_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    if df.empty or "pre_pyrolysis_method_group" not in df.columns:
        return ranges
    for method, group in df.groupby("pre_pyrolysis_method_group"):
        temp = pd.to_numeric(group["pre_pyrolysis_temp_c"], errors="coerce").dropna()
        time = pd.to_numeric(group["pre_pyrolysis_time_h"], errors="coerce").dropna()
        ranges[str(method)] = {
            "temp_min": float(temp.min()) if not temp.empty else 0.0,
            "temp_median": float(temp.median()) if not temp.empty else 0.0,
            "temp_max": float(temp.max()) if not temp.empty else 0.0,
            "time_min": float(time.min()) if not time.empty else 0.0,
            "time_median": float(time.median()) if not time.empty else 0.0,
            "time_max": float(time.max()) if not time.empty else 0.0,
            "n": int(len(group)),
        }
    return ranges


def friendly_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 100 or float(value).is_integer():
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def safe_numeric_series(values: pd.Series | list | np.ndarray) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna()


def safe_series_median(values: pd.Series | list | np.ndarray, fallback: float = np.nan) -> float:
    s = safe_numeric_series(values)
    if s.empty:
        return float(fallback)
    return float(s.median())


def safe_series_quantile(values: pd.Series | list | np.ndarray, q: float, fallback: float = np.nan) -> float:
    s = safe_numeric_series(values)
    if s.empty:
        return float(fallback)
    return float(s.quantile(q))


def safe_series_std(values: pd.Series | list | np.ndarray, fallback: float = 1.0) -> float:
    s = safe_numeric_series(values)
    if len(s) < 2:
        return float(fallback)
    value = float(s.std())
    if not np.isfinite(value) or value == 0:
        return float(fallback)
    return value


def safe_column_medians(df: pd.DataFrame, cols: list[str], fallback: pd.Series | None = None) -> pd.Series:
    fallback = pd.Series(dtype=float) if fallback is None else fallback
    values = {}
    for col in cols:
        values[col] = safe_series_median(df[col], fallback=float(fallback.get(col, np.nan))) if col in df.columns else float(fallback.get(col, np.nan))
    return pd.Series(values, dtype=float)


def safe_column_scales(df: pd.DataFrame, cols: list[str], fallback: float = 1.0) -> pd.Series:
    return pd.Series(
        {col: safe_series_std(df[col], fallback=fallback) if col in df.columns else fallback for col in cols},
        dtype=float,
    )


def range_caption(label: str, value: float, stats: dict, unit: str = "") -> None:
    unit_text = f" {unit}" if unit else ""
    if stats["min"] <= value <= stats["max"]:
        cls = "range-ok"
        status = "inside training range"
    else:
        cls = "range-warn"
        status = "outside training range"
    st.markdown(
        f"<span class='{cls}'>{label}: {status}</span><br>"
        f"<span class='small-muted'>Training range {friendly_number(stats['min'])}-{friendly_number(stats['max'])}{unit_text}; "
        f"common range {friendly_number(stats['q05'])}-{friendly_number(stats['q95'])}{unit_text}.</span>",
        unsafe_allow_html=True,
    )


def method_to_raw_text(method: str) -> str:
    return {
        "none": "none",
        "sol_blending": "sol_blending",
        "sol_gel": "sol_gel",
        "sol_gel_thermal": "sol_gel_thermal",
        "uv_crosslinking": "uv_crosslinking",
        "thermal_crosslinking": "thermal_crosslinking",
        "catalyzed_crosslinking": "catalyzed_crosslinking",
        "staged_pyrolysis": "staged_pyrolysis",
        "solvothermal_autoclave": "solvothermal_autoclave",
    }[method]


def ensure_dvb_text(polymer: str, dvb_present: bool, dvb_ratio: float | None = None) -> str:
    text = polymer.strip() if polymer else "custom precursor"
    has_dvb = any(token in text.lower() for token in ["dvb", "divinylbenzene", "divinylbenzen"])
    if dvb_present and not has_dvb:
        if dvb_ratio is not None and dvb_ratio > 0:
            return f"{text} with DVB {dvb_ratio:.3g}:1"
        return f"{text} with DVB"
    if not dvb_present and has_dvb:
        return text.replace("DVB", "").replace("dvb", "").replace("Divinylbenzene", "").replace("divinylbenzene", "")
    return text


@st.cache_data(show_spinner=False)
def template_family_lookup() -> dict[str, str]:
    labels = [label for label in POLYMER_FAMILY_TEMPLATES if label != "Other / custom"]
    templates = [POLYMER_FAMILY_TEMPLATES[label] for label in labels]
    engineered = prepare_features(pd.DataFrame({"polymer": templates}))
    lookup = dict(zip(labels, engineered["polymer_family_broad"].astype(str)))
    lookup["Other / custom"] = "other"
    return lookup


def template_family(label: str) -> str:
    return template_family_lookup().get(label, "other")


N_ADDITIVE_PATTERN = r"pvp|polyvinylpyrrolidone|pyrrole"
N_ADDITIVE_LABELS = {
    "Phenylalkyl polysiloxane + pyrrole/PVP",
    "Vinyl polysiloxane + PVP",
}
INTRINSIC_N_SOURCE_LABELS = {
    "Polysilazane / polyorganosilazane",
    "Silylcarbodiimide / silylsesquiazane",
}
POLYSILOXANE_MODEL_FAMILIES = {
    "phenylalkyl_polysiloxane",
    "alkyl_polysiloxane",
    "vinyl_polysiloxane",
    "phenyl_polysiloxane",
    "polysiloxane",
}


def label_specific_family_subset(label: str, family: str, ref: pd.DataFrame) -> pd.DataFrame:
    subset = ref[ref["polymer_family_broad"].astype(str) == family].copy()
    polymer_text = subset.get("polymer", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    additive_n = polymer_text.str.contains(N_ADDITIVE_PATTERN, regex=True, na=False)
    if label in N_ADDITIVE_LABELS:
        return subset[additive_n].copy()
    if family in POLYSILOXANE_MODEL_FAMILIES:
        return subset[~additive_n].copy()
    return subset


def family_composition_match_table(
    si: float,
    c: float,
    o: float,
    n: float,
    engineered_ref: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    composition_cols = ["si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct"]
    family_lookup = template_family_lookup()
    if engineered_ref.empty or "polymer_family_broad" not in engineered_ref.columns:
        for label in POLYMER_FAMILY_TEMPLATES:
            family = family_lookup.get(label, "other")
            if label == "Other / custom":
                match_pct = 1.0
                reason = "Custom precursor is kept as a manual escape hatch."
            elif n >= 3.0 and label in INTRINSIC_N_SOURCE_LABELS:
                match_pct = 75.0
                reason = "No public reference table loaded; intrinsic N-source family kept for high-N targets."
            elif 0.3 <= n < 3.0 and label in N_ADDITIVE_LABELS:
                match_pct = 75.0
                reason = "No public reference table loaded; explicit PVP/pyrrole low-N additive route kept."
            elif n >= 0.3 and family in POLYSILOXANE_MODEL_FAMILIES and label not in N_ADDITIVE_LABELS:
                match_pct = 8.0
                reason = "No public reference table loaded; plain polysiloxane has no explicit N source."
            elif n < 0.3 and label in INTRINSIC_N_SOURCE_LABELS:
                match_pct = 10.0
                reason = "No public reference table loaded; N-source family penalized for N-free target."
            else:
                match_pct = 50.0
                reason = "No public reference table loaded; using conservative default family match."
            rows.append({
                "label": label,
                "family": family,
                "match_pct": match_pct,
                "closest_distance": np.nan,
                "n_samples": 0,
                "reason": reason,
            })
        return pd.DataFrame(rows).sort_values(["match_pct", "label"], ascending=[False, True]).reset_index(drop=True)

    ref = engineered_ref.dropna(subset=["polymer_family_broad"]).copy()
    target = pd.Series({"si_wt_pct": si, "c_wt_pct": c, "o_wt_pct": o, "n_wt_pct": n})
    global_scale = safe_column_scales(ref, composition_cols, fallback=1.0)
    global_median = safe_column_medians(ref, composition_cols)

    for label in POLYMER_FAMILY_TEMPLATES:
        family = family_lookup.get(label, "other")
        if label == "Other / custom":
            rows.append({
                "label": label,
                "family": family,
                "match_pct": 1.0,
                "closest_distance": np.nan,
                "n_samples": 0,
                "reason": "Custom precursor is kept as a manual escape hatch.",
            })
            continue

        subset = label_specific_family_subset(label, family, ref)
        if subset.empty:
            rows.append({
                "label": label,
                "family": family,
                "match_pct": 0.0,
                "closest_distance": np.nan,
                "n_samples": 0,
                "reason": "No cleaned-dataset rows currently map to this family.",
            })
            continue

        med = safe_column_medians(subset, composition_cols, fallback=global_median)
        vals = subset[composition_cols].apply(pd.to_numeric, errors="coerce").fillna(med)
        diff = (vals - target) / global_scale
        distances = np.sqrt((diff**2).sum(axis=1))
        closest_distance = float(distances.min())
        if not np.isfinite(closest_distance):
            closest_distance = np.nan
            match_pct = 0.0
        else:
            match_pct = 100.0 / (1.0 + max(closest_distance, 0.0))

        q05 = pd.Series({
            col: safe_series_quantile(subset[col], 0.05, fallback=float(med[col]))
            for col in composition_cols
        })
        q95 = pd.Series({
            col: safe_series_quantile(subset[col], 0.95, fallback=float(med[col]))
            for col in composition_cols
        })
        inside = [col for col in composition_cols if q05[col] <= target[col] <= q95[col]]
        reason = f"{len(inside)}/4 elements inside this family common range; {len(subset)} literature row(s)."

        if n >= 3.0 and label in N_ADDITIVE_LABELS:
            match_pct = min(match_pct, 12.0)
            reason = "Low-N additive route only: PVP/pyrrole examples do not cover high-N SiCN-like compositions."
        elif 0.3 <= n < 3.0 and family in POLYSILOXANE_MODEL_FAMILIES and label not in N_ADDITIVE_LABELS:
            match_pct = min(match_pct, 8.0)
            reason = "Plain polysiloxane has no N source; choose an explicit PVP/pyrrole additive route for low-N targets."
        elif n >= 1.0 and label not in N_ADDITIVE_LABELS and label not in INTRINSIC_N_SOURCE_LABELS and float(q95["n_wt_pct"]) < 0.5:
            match_pct = min(match_pct, 5.0)
            reason = "Filtered by N gate: this family has no meaningful N-containing examples."
        elif n < 0.5 and float(q05["n_wt_pct"]) >= 0.5:
            match_pct = min(match_pct, 10.0)
            reason = "Filtered by N gate: this family is mainly N-containing."

        rows.append({
            "label": label,
            "family": family,
            "match_pct": match_pct,
            "closest_distance": closest_distance,
            "n_samples": int(len(subset)),
            "reason": reason,
        })

    return pd.DataFrame(rows).sort_values(["match_pct", "n_samples"], ascending=[False, False]).reset_index(drop=True)


def compatible_polymer_options(
    si: float,
    c: float,
    o: float,
    n: float,
    engineered_ref: pd.DataFrame,
    min_match_pct: float,
) -> tuple[list[str], pd.DataFrame]:
    match_table = family_composition_match_table(si, c, o, n, engineered_ref)
    options = match_table.loc[match_table["match_pct"] >= min_match_pct, "label"].tolist()
    options = [label for label in options if label != "Other / custom"]
    if not options:
        options = match_table.loc[match_table["label"] != "Other / custom", "label"].head(3).tolist()
    options.append("Other / custom")
    return options, match_table


def polymer_dvb_allowed(family: str, engineered_ref: pd.DataFrame) -> tuple[bool, str]:
    if family == "Other / custom":
        return True, "Custom precursor: enable DVB only when the chemistry can copolymerize/crosslink with divinylbenzene."
    template = POLYMER_FAMILY_TEMPLATES.get(family, "")
    test = prepare_features(pd.DataFrame([{"polymer": template}]))
    polymer_family_broad = str(test.loc[0, "polymer_family_broad"])
    if engineered_ref.empty or "dvb_modification" not in engineered_ref.columns:
        fallback = any(token in family.lower() for token in ["siloxane", "silazane", "vinyl", "rd-684", "htt1800"])
        return fallback, "DVB compatibility is estimated from precursor chemistry because reference data are unavailable."

    if polymer_family_broad != "other":
        subset = engineered_ref[engineered_ref["polymer_family_broad"].astype(str) == polymer_family_broad]
    else:
        family_tokens = [t for t in template.lower().replace("/", " ").split() if len(t) >= 4]
        polymer_text = engineered_ref.get("polymer", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
        mask = pd.Series(False, index=engineered_ref.index)
        for token in family_tokens:
            mask = mask | polymer_text.str.contains(token, regex=False)
        subset = engineered_ref[mask]

    dvb_count = int(pd.to_numeric(subset.get("dvb_modification", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    if dvb_count > 0:
        return True, f"DVB enabled: {dvb_count} related cleaned-dataset row(s) contain DVB for this precursor type."
    return False, "DVB disabled: no related cleaned-dataset polymer rows contain DVB for this precursor type."


def build_polymer_text(si: float, c: float, o: float, n: float, engineered_ref: pd.DataFrame) -> tuple[str, dict, bool, float]:
    with st.sidebar.expander("Polymer / precursor chemistry", expanded=True):
        min_family_match_pct = st.slider(
            "Minimum family match (%)",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
        help="Filters precursor families by elemental Si/C/O/N composition match only.",
        )
        options, match_table = compatible_polymer_options(
            si, c, o, n, engineered_ref, min_family_match_pct
        )
        family = st.selectbox(
            "Polymer / precursor type",
            options,
            index=0,
            help="Filtered by elemental-composition match. The selected family guides recipe chemistry; it is not used directly by the prediction model.",
        )
        selected_match = match_table[match_table["label"] == family]
        if not selected_match.empty:
            st.caption(
                f"Composition match: {float(selected_match['match_pct'].iloc[0]):.0f}% | "
                f"{selected_match['reason'].iloc[0]}"
            )
        st.caption(POLYMER_COMPATIBILITY.get(family, "Use as an exploratory precursor type."))
        with st.expander("Why these precursor choices?", expanded=False):
            display = match_table[["label", "match_pct", "n_samples", "reason"]].copy()
            display = display[display["label"] != "Other / custom"].head(12)
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={
                    "label": "precursor family",
                    "match_pct": st.column_config.NumberColumn("match", format="%.0f%%"),
                    "n_samples": st.column_config.NumberColumn("rows", format="%d"),
                    "reason": "basis",
                },
            )

        custom_note = ""
        if family == "Other / custom":
            custom_note = st.text_input("Custom precursor note", value="custom precursor")

        base = custom_note.strip() if custom_note else POLYMER_FAMILY_TEMPLATES[family]
        dvb_allowed, dvb_tip = polymer_dvb_allowed(family, engineered_ref)

        dvb_present = st.checkbox(
            "DVB modification",
            value=False,
            help="DVB is retained as synthesis/recipe context. It is not used directly by the final capacity prediction model.",
            disabled=not dvb_allowed,
        )
        dvb_ratio = st.number_input(
            "DVB:base wt. ratio",
            min_value=0.0,
            max_value=5.0,
            value=1.0 if dvb_present else 0.0,
            step=0.1,
            disabled=(not dvb_present) or (not dvb_allowed),
        )
        st.caption(dvb_tip)

        polymer = base.strip()
        st.caption("The app uses this precursor name for recipe guidance and to parse DVB loading; polymer family is not an active prediction feature.")
        return polymer, {"polymer_family_ui": family, "polymer_tip_ui": POLYMER_COMPATIBILITY.get(family, "")}, dvb_present, dvb_ratio


def phase_progress(label: str, value: float, help_text: str) -> None:
    value = 0.0 if pd.isna(value) else float(value)
    st.write(f"**{label}:** {value:.3f}")
    st.progress(min(max(value, 0.0), 1.0), text=help_text)


def phase_formula_summary(engineered: pd.DataFrame) -> dict[str, float | str]:
    row = engineered.iloc[0]
    sio2 = max(float(row.get("sio2_phase", 0) or 0), 0.0)
    sic = max(float(row.get("sic_phase", 0) or 0), 0.0)
    free_c = max(float(row.get("free_c_phase", 0) or 0), 0.0)
    si3n4 = max(float(row.get("si3n4_phase", 0) or 0), 0.0)
    phase_total = sio2 + sic + free_c + si3n4
    if phase_total <= 0:
        phase_total = np.nan

    x_si = float(row.get("x_si", np.nan))
    y_c_atomic = float(row.get("y_c", np.nan))
    z_o_atomic = float(row.get("z_o", np.nan))
    w_n_atomic = float(row.get("w_n", np.nan))
    si_for_nitride = min(x_si, 0.75 * w_n_atomic) if np.isfinite(x_si) and np.isfinite(w_n_atomic) else 0.0
    si_matrix = max(x_si - si_for_nitride, 1e-9) if np.isfinite(x_si) else 1e-9
    total_o_per_si = max(z_o_atomic / x_si, 0.0) if np.isfinite(z_o_atomic) and np.isfinite(x_si) and x_si > 0 else np.nan
    total_c_per_si = max(y_c_atomic / x_si, 0.0) if np.isfinite(y_c_atomic) and np.isfinite(x_si) and x_si > 0 else np.nan
    total_n_per_si = max(w_n_atomic / x_si, 0.0) if np.isfinite(w_n_atomic) and np.isfinite(x_si) and x_si > 0 else np.nan
    o_per_si = max(z_o_atomic / si_matrix, 0.0) if np.isfinite(z_o_atomic) else np.nan
    c_total_per_si = max(y_c_atomic / si_matrix, 0.0) if np.isfinite(y_c_atomic) else np.nan
    # Oxygen consumes Si as SiO2-like units; the remaining Si can form SiC-like C.
    c_bound_per_si = min(c_total_per_si, max(1.0 - (o_per_si / 2.0), 0.0)) if np.isfinite(o_per_si) and np.isfinite(c_total_per_si) else np.nan
    c_free_per_si = max(c_total_per_si - c_bound_per_si, 0.0) if np.isfinite(c_bound_per_si) and np.isfinite(c_total_per_si) else np.nan

    def html_formula(
        o_value: float,
        c_value: float,
        n_value: float | None = None,
        free_value: float | None = None,
        nitride_fraction: float | None = None,
    ) -> str:
        if not np.isfinite(o_value) or not np.isfinite(c_value):
            return "unavailable"
        base = f"SiO<sub>{o_value:.2f}</sub>C<sub>{c_value:.2f}</sub>"
        if n_value is not None and np.isfinite(n_value) and n_value > 0.005:
            base = f"{base}N<sub>{n_value:.2f}</sub>"
        if free_value is not None and np.isfinite(free_value):
            base = f"{base}.(C<sub>free</sub>)<sub>{free_value:.2f}</sub>"
        if nitride_fraction is not None and np.isfinite(nitride_fraction) and nitride_fraction > 0.005:
            base = f"{base}.(Si<sub>3</sub>N<sub>4</sub>)<sub>{nitride_fraction:.2f}</sub>"
        return base

    return {
        "sio2_ratio": sio2 / phase_total if np.isfinite(phase_total) else np.nan,
        "sic_ratio": sic / phase_total if np.isfinite(phase_total) else np.nan,
        "free_c_ratio": free_c / phase_total if np.isfinite(phase_total) else np.nan,
        "si3n4_ratio": si3n4 / phase_total if np.isfinite(phase_total) else np.nan,
        "sioxy_total_formula": html_formula(total_o_per_si, total_c_per_si, total_n_per_si),
        "sioxy_bound_formula": html_formula(o_per_si, c_bound_per_si, nitride_fraction=si3n4),
        "sioxy_freec_formula": html_formula(o_per_si, c_bound_per_si, free_value=c_free_per_si, nitride_fraction=si3n4),
    }


def render_phase_dashboard(engineered_input: pd.DataFrame) -> None:
    summary = phase_formula_summary(engineered_input)
    st.subheader("Composition-derived phase proxies")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        phase_progress("SiO₂-like", engineered_input.loc[0, "sio2_phase"], "oxide-like network proxy")
    with p2:
        phase_progress("SiC-like", engineered_input.loc[0, "sic_phase"], "carbide-like matrix proxy")
    with p3:
        phase_progress("Free carbon", engineered_input.loc[0, "free_c_phase"], "conductive network proxy")
    with p4:
        phase_progress("Si₃N₄-like", engineered_input.loc[0, "si3n4_phase"], "N-containing network proxy")

    st.subheader("Normalized phase ratios and formulas")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("SiO₂", "n/a" if pd.isna(summary["sio2_ratio"]) else f"{100 * summary['sio2_ratio']:.1f}%")
    r2.metric("SiC", "n/a" if pd.isna(summary["sic_ratio"]) else f"{100 * summary['sic_ratio']:.1f}%")
    r3.metric("Free C", "n/a" if pd.isna(summary["free_c_ratio"]) else f"{100 * summary['free_c_ratio']:.1f}%")
    r4.metric("Si₃N₄", "n/a" if pd.isna(summary["si3n4_ratio"]) else f"{100 * summary['si3n4_ratio']:.1f}%")
    st.markdown(
        f"<span class='formula-pill'>Total: {summary['sioxy_total_formula']}</span>"
        f"<span class='formula-pill'>Matrix: {summary['sioxy_bound_formula']}</span>"
        f"<span class='formula-pill'>Split: {summary['sioxy_freec_formula']}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Percentages are normalized mole-basis phase-proxy fractions derived from elemental atomic ratios, not wt.%. "
        "The raw phase columns below are unnormalized formula-unit proxies, so they do not sum to 1 until divided by their phase total."
    )


def build_raw_input(reference: pd.DataFrame, engineered_ref: pd.DataFrame, selected_target: str, model_features: list[str]) -> tuple[pd.DataFrame, dict]:
    ranges = {col: get_numeric_range(engineered_ref, col, fallback) for col, fallback in {
        "si_wt_pct": (0.0, 32.0, 62.0),
        "c_wt_pct": (0.0, 43.0, 100.0),
        "o_wt_pct": (0.0, 21.0, 76.0),
        "n_wt_pct": (0.0, 0.0, 26.0),
        "pyrolysis_temp_c": (600.0, 1000.0, 2000.0),
        "pyrolysis_time_h": (0.5, 1.0, 7.0),
        "cycling_numbers": (50.0, 100.0, 200.0),
    }.items()}
    with st.sidebar:
        st.header("Material Input")
        st.caption("Input ranges are bounded by the cleaned training data. Common ranges show the 5th-95th percentile.")

        with st.expander("Composition", expanded=True):
            si = st.number_input("Si wt.%", min_value=ranges["si_wt_pct"]["min"], max_value=ranges["si_wt_pct"]["max"], value=35.0, step=0.1)
            c = st.number_input("C wt.%", min_value=ranges["c_wt_pct"]["min"], max_value=ranges["c_wt_pct"]["max"], value=45.0, step=0.1)
            o = st.number_input("O wt.%", min_value=ranges["o_wt_pct"]["min"], max_value=ranges["o_wt_pct"]["max"], value=18.0, step=0.1)
            n = st.number_input("N wt.%", min_value=ranges["n_wt_pct"]["min"], max_value=ranges["n_wt_pct"]["max"], value=0.0, step=0.1)
            total = si + c + o + n
            if abs(total - 100) > 1.0:
                st.warning(f"Si + C + O + N = {total:.1f} wt.%. The model was trained mostly on compositions close to 100 wt.%")

        pre_method = "none"
        pre_temp = np.nan
        pre_time = np.nan
        pre_atm = "unknown"

        with st.expander("Pyrolysis", expanded=True):
            pyro_temp = st.number_input(
                "Pyrolysis temperature (C)",
                min_value=int(ranges["pyrolysis_temp_c"]["min"]),
                max_value=int(ranges["pyrolysis_temp_c"]["max"]),
                value=1000,
                step=50,
                help="Final ceramic conversion temperature. Bounded by training data.",
            )
            pyro_time = st.number_input(
                "Pyrolysis time (h)",
                min_value=float(ranges["pyrolysis_time_h"]["min"]),
                max_value=float(ranges["pyrolysis_time_h"]["max"]),
                value=1.0,
                step=0.5,
            )

        pyro_atm = "inert"
        with st.expander("Pyrolysis atmosphere context", expanded=False):
            st.info(
                "Pyrolysis atmosphere is not used by the deployed model. "
                "In the cleaned dataset, 212/219 first-capacity rows (~97%) are interpreted as primary inert/protective pyrolysis "
                "(Ar, N2, Ar/H2, reported inert, unknown, or Ar pyrolysis followed by air treatment), "
                "so the prediction is interpreted for inert-atmosphere pyrolysis."
            )

        first_current = LOW_RATE_FIRST_CURRENT_MA_G
        cycling_current = LOW_RATE_CYCLING_CURRENT_MA_G
        with st.expander("Capacity context", expanded=True):
            st.info(
                "Prediction is interpreted for low-current literature testing, about "
                "0.05-0.1C graphite-equivalent (18.6-37.2 mA/g using 372 mAh/g graphite) "
                "and a 0-3 V voltage window."
            )
            cycling_numbers = st.selectbox(
                "Cycle number for cyclability",
                [50, 100, 200],
                index=1,
                help="Constrained to common literature endpoints. Most stable/cycled rows are concentrated between 50 and 200 cycles.",
            )
            st.caption("Current density and voltage window are not used by the deployed prediction models.")

        with st.expander("First-cycle CE options", expanded=False):
            ce_mode = st.radio(
                "CE estimate",
                ["Diagnostic CE using predicted Qrev", "Design-only CE", "Diagnostic CE using measured Qrev"],
                horizontal=False,
                help=(
                    "Design-only CE uses elemental composition and pyrolysis conditions. "
                    "Diagnostic CE adds first-cycle reversible capacity (`reversible_capacity_mah_g`) because CE is linked to Qrev and first-cycle loss. "
                    "Use predicted Qrev for virtual screening; use measured Qrev after an experiment for a more diagnostic estimate."
                ),
            )
            measured_qrev = st.number_input(
                "Measured first-cycle Qrev (mAh/g)",
                min_value=0.0,
                max_value=3000.0,
                value=500.0,
                step=10.0,
                disabled=(ce_mode != "Diagnostic CE using measured Qrev"),
                help=(
                    "Optional experimental first-cycle reversible capacity. "
                    "This is used only when 'Diagnostic CE using measured Qrev' is selected. "
                    "Otherwise the app uses the model-predicted Qrev or the design-only CE model."
                ),
            )
            st.caption("Default uses predicted Qrev as a diagnostic input. Design-only CE remains available for strict pre-test screening.")

        with st.expander("Cyclability options", expanded=False):
            stable_mode = st.radio(
                "Cyclability estimate",
                ["Diagnostic Qcycled using predicted Qrev", "Design-only Qcycled", "Diagnostic Qcycled using measured Qrev"],
                horizontal=False,
                help=(
                    "Design-only Qcycled predicts cycled capacity from material design plus selected cycle number. "
                    "Diagnostic Qcycled additionally uses first-cycle reversible capacity (`reversible_capacity_mah_g`), "
                    "which improves the stable/cycled capacity estimate after first-cycle behavior is known. "
                    "Use predicted Qrev for screening; use measured Qrev after first-cycle testing."
                ),
            )
            measured_qrev_stable = st.number_input(
                "Measured first-cycle Qrev for cyclability (mAh/g)",
                min_value=0.0,
                max_value=3000.0,
                value=500.0,
                step=10.0,
                disabled=(stable_mode != "Diagnostic Qcycled using measured Qrev"),
                help=(
                    "Optional experimental first-cycle reversible capacity used by the diagnostic cyclability model. "
                    "It is ignored unless 'Diagnostic Qcycled using measured Qrev' is selected."
                ),
            )
            st.caption("Default uses predicted Qrev as a diagnostic input. Design-only Qcycled remains available for strict pre-test screening.")

        surface_area = np.nan
        use_surface_assisted = False
        with st.expander("Measured structure context", expanded=False):
            use_surface_assisted = st.checkbox(
                "Use optional surface-assisted model",
                value=False,
                help="Use only when BET surface area is measured or intentionally specified for a sensitivity check.",
            )
            surface_stats = get_numeric_range(engineered_ref, "surface_area_m2_g", (0.0, 100.0, 1500.0))
            surface_area = st.number_input(
                "BET surface area (m2/g)",
                min_value=0.0,
                max_value=max(surface_stats["max"], 1.0),
                value=float(surface_stats["median"]) if use_surface_assisted else 0.0,
                step=10.0,
                disabled=not use_surface_assisted,
            )
            st.info(
                "The main model does not require surface area. The optional surface-assisted model uses raw BET surface area "
                "and should be interpreted as a characterized-material or sensitivity estimate."
            )

    raw_input = pd.DataFrame([{
        "polymer": "target composition only",
        "si_wt_pct": si,
        "c_wt_pct": c,
        "o_wt_pct": o,
        "n_wt_pct": n,
        "pyrolysis_temp_c": pyro_temp,
        "pyrolysis_time_h": pyro_time,
        "pyrolysis_atmosphere": pyro_atm,
        "pre_pyrolysis_method": method_to_raw_text(pre_method),
        "crosslinking_method": method_to_raw_text(pre_method),
        "crosslink_temp_c": np.nan if pre_method == "none" else pre_temp,
        "crosslink_time_h": np.nan if pre_method == "none" else pre_time,
        "crosslink_atmosphere": pre_atm,
        "first_cycling_current_ma_g": first_current,
        "cycling_numbers": cycling_numbers,
        "cycling_current_ma_g": cycling_current,
        "surface_area_m2_g": surface_area,
        "use_surface_assisted_ui": use_surface_assisted,
        "voltage_min_v": 0.0,
        "voltage_max_v": 3.0,
        "voltage_window_v": 3.0,
        "ce_mode_ui": ce_mode,
        "stable_mode_ui": stable_mode,
        "measured_qrev_ui": measured_qrev,
        "measured_qrev_stable_ui": measured_qrev_stable,
    }])
    return raw_input, ranges


def feature_matrix(engineered: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    cols = list(bundle.get("feature_columns", []))
    for col in cols:
        if col not in engineered.columns:
            engineered[col] = np.nan
    return engineered[cols].copy()


def nearest_literature_rows(raw_input: pd.DataFrame, reference: pd.DataFrame, target: str, n: int = 8) -> pd.DataFrame:
    if reference.empty:
        return pd.DataFrame()
    ref = prepare_features(reference)
    inp = prepare_features(raw_input)
    target_n = float(inp.loc[0, "n_wt_pct"]) if "n_wt_pct" in inp.columns else 0.0
    if target_n >= 1.0 and "n_wt_pct" in ref.columns:
        n_ref = ref[pd.to_numeric(ref["n_wt_pct"], errors="coerce").fillna(0) > 0].copy()
        if len(n_ref) >= min(n, 5):
            ref = n_ref
    cols = ["si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct"]
    cols = [c for c in cols if c in ref.columns and c in inp.columns]
    if not cols:
        return pd.DataFrame()
    med = safe_column_medians(ref, cols)
    ref_vals = ref[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    inp_vals = inp.loc[[0], cols].fillna(med)
    scale = safe_column_scales(ref_vals, cols, fallback=1.0)
    diff = (ref_vals - inp_vals.iloc[0]) / scale
    ref["elemental_composition_distance"] = np.sqrt((diff**2).sum(axis=1))
    ref["composition_match_pct"] = 100.0 / (1.0 + ref["elemental_composition_distance"].clip(lower=0))
    keep = [
        "polymer", "polymer_family_broad", "precursor_family",
        "si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct",
        "pyrolysis_temp_c", "pyrolysis_time_h",
        "cycling_numbers", TARGET, STABLE_TARGET,
        "sample_count", "elemental_composition_distance", "composition_match_pct",
        "reference", "doi",
    ]
    keep = [c for c in keep if c in ref.columns]
    out = ref.sort_values("elemental_composition_distance").head(n)[keep].copy()
    if "doi" in out.columns:
        out["doi_link"] = out["doi"].apply(normalize_doi)
    return out


def plot_prediction_context(reference: pd.DataFrame, target: str, pred: float) -> None:
    s = pd.to_numeric(reference.get(target, pd.Series(dtype=float)), errors="coerce").dropna()
    if s.empty:
        st.info("Training/reference distribution is unavailable in this deployment.")
        return
    weights = None
    if "sample_count" in reference.columns:
        weights = pd.to_numeric(reference.loc[s.index, "sample_count"], errors="coerce").fillna(1.0).clip(lower=1.0)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(s, bins=min(24, max(8, len(s))), weights=weights, color="#5b8fb9", alpha=0.78, edgecolor="white")
    ax.axvline(pred, color="#b42318", linewidth=2.2, label=f"Prediction: {pred:.0f}")
    ax.axvline(s.median(), color="#333333", linestyle="--", linewidth=1.3, label=f"Training median: {s.median():.0f}")
    ax.set_xlabel("Capacity (mAh/g)")
    ax.set_ylabel("Weighted count" if weights is not None else "Count")
    ax.legend()
    title = "Prediction relative to public aggregate reference distribution" if weights is not None else "Prediction relative to training distribution"
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


def model_summary(bundle: dict | None) -> str:
    if bundle is None:
        return "Missing model bundle"
    deployed = bundle.get("deployed_feature_set")
    suffix = f" | {deployed}" if deployed else ""
    target_label = bundle.get("label") or bundle.get("target", "target")
    return f"{target_label}<br>{bundle.get('best_model_name', 'model')} | {bundle.get('n_training_rows', '?')} rows | {bundle.get('n_groups', '?')} groups{suffix}"


def graphite_comparison(capacity: float, graphite_capacity: float = 372.0) -> str:
    delta = capacity - graphite_capacity
    ratio = capacity / graphite_capacity if graphite_capacity else np.nan
    if delta >= 0:
        return f"{delta:.0f} mAh/g above graphite, about {ratio:.2f}x graphite's theoretical capacity ({graphite_capacity:.0f} mAh/g)."
    return f"{abs(delta):.0f} mAh/g below graphite, about {ratio:.2f}x graphite's theoretical capacity ({graphite_capacity:.0f} mAh/g)."


def build_composition_input(reference: pd.DataFrame, engineered_ref: pd.DataFrame, selected_target: str) -> tuple[pd.DataFrame, dict]:
    ranges = {col: get_numeric_range(engineered_ref, col, fallback) for col, fallback in {
        "si_wt_pct": (0.0, 32.0, 62.0),
        "c_wt_pct": (0.0, 43.0, 100.0),
        "o_wt_pct": (0.0, 21.0, 76.0),
        "n_wt_pct": (0.0, 0.0, 26.0),
        "pyrolysis_temp_c": (600.0, 1000.0, 2000.0),
        "pyrolysis_time_h": (0.5, 1.0, 7.0),
        "cycling_numbers": (50.0, 100.0, 200.0),
    }.items()}

    with st.sidebar:
        st.header("Target Composition")
        st.caption("Enter the desired elemental composition. The app searches literature-like routes that could plausibly reach nearby Si-O-C-N chemistry.")
        si = st.number_input("Target Si wt.%", min_value=ranges["si_wt_pct"]["min"], max_value=ranges["si_wt_pct"]["max"], value=35.0, step=0.1)
        c = st.number_input("Target C wt.%", min_value=ranges["c_wt_pct"]["min"], max_value=ranges["c_wt_pct"]["max"], value=45.0, step=0.1)
        o = st.number_input("Target O wt.%", min_value=ranges["o_wt_pct"]["min"], max_value=ranges["o_wt_pct"]["max"], value=18.0, step=0.1)
        n = st.number_input("Target N wt.%", min_value=ranges["n_wt_pct"]["min"], max_value=ranges["n_wt_pct"]["max"], value=0.0, step=0.1)
        total = si + c + o + n
        if abs(total - 100) > 1.0:
            st.warning(f"Si + C + O + N = {total:.1f} wt.%. Normalize or interpret this as an approximate target.")

        st.header("Route Search")
        n_neighbors = st.slider("Similar literature samples", min_value=8, max_value=40, value=20, step=2)
        route_min_match_pct = st.slider(
            "Minimum route-family match (%)",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            help="Filters suggested recipe families using the same composition-match score as manual mode.",
        )
        current_context = LOW_RATE_FIRST_CURRENT_MA_G
        cycling_current = LOW_RATE_CYCLING_CURRENT_MA_G
        st.info(
            "Route search is synthesis guidance only. It does not change the ML prediction, which is interpreted "
            "at low-current 0.05-0.1C graphite-equivalent and a 0-3 V voltage window."
        )
        cycling_numbers = st.selectbox(
            "Cycle number for cyclability context",
            [50, 100, 200],
            index=1,
        )

    raw_input = pd.DataFrame([{
        "polymer": "target composition only",
        "si_wt_pct": si,
        "c_wt_pct": c,
        "o_wt_pct": o,
        "n_wt_pct": n,
        "pyrolysis_temp_c": ranges["pyrolysis_temp_c"]["median"],
        "pyrolysis_time_h": ranges["pyrolysis_time_h"]["median"],
        "pyrolysis_atmosphere": "inert",
        "pre_pyrolysis_method": "none",
        "crosslinking_method": "none",
        "crosslink_temp_c": np.nan,
        "crosslink_time_h": np.nan,
        "crosslink_atmosphere": "unknown",
        "first_cycling_current_ma_g": current_context,
        "cycling_numbers": cycling_numbers,
        "cycling_current_ma_g": cycling_current,
        "surface_area_m2_g": np.nan,
        "use_surface_assisted_ui": False,
        "voltage_min_v": 0.0,
        "voltage_max_v": 3.0,
        "voltage_window_v": 3.0,
        "ce_mode_ui": "Diagnostic CE using predicted Qrev",
        "stable_mode_ui": "Diagnostic Qcycled using predicted Qrev",
        "measured_qrev_ui": np.nan,
        "measured_qrev_stable_ui": np.nan,
    }])
    return raw_input, {
        **ranges,
        "n_neighbors": n_neighbors,
        "route_min_match_pct": route_min_match_pct,
    }


def composition_neighbors(target_engineered: pd.DataFrame, engineered_ref: pd.DataFrame, n: int) -> pd.DataFrame:
    if engineered_ref.empty:
        return pd.DataFrame()
    target_n = float(target_engineered.loc[0, "n_wt_pct"]) if "n_wt_pct" in target_engineered.columns else 0.0
    ref = engineered_ref.copy()
    if target_n >= 1.0 and "n_wt_pct" in ref.columns:
        n_ref = ref[pd.to_numeric(ref["n_wt_pct"], errors="coerce").fillna(0) > 0].copy()
        if len(n_ref) >= min(n, 5):
            ref = n_ref
    cols = ["si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct"]
    cols = [c for c in cols if c in engineered_ref.columns and c in target_engineered.columns]
    if not cols:
        return pd.DataFrame()
    med = safe_column_medians(ref, cols)
    ref_vals = ref[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    target_vals = target_engineered.loc[[0], cols].fillna(med)
    scale = safe_column_scales(ref_vals, cols, fallback=1.0)
    diff = (ref_vals - target_vals.iloc[0]) / scale
    ref["elemental_composition_distance"] = np.sqrt((diff**2).sum(axis=1))
    ref["composition_match_pct"] = 100.0 / (1.0 + ref["elemental_composition_distance"].clip(lower=0))
    return ref.sort_values("elemental_composition_distance").head(n).copy()


def route_label(row: pd.Series) -> str:
    dvb = "DVB-modified" if int(row.get("dvb_modification", 0) or 0) else "no DVB"
    return f"{row.get('polymer_family_broad', row.get('precursor_family', 'unknown'))} | {dvb}"


def representative_polymer(precursor_family: str, dvb_modification: int) -> str:
    family_text = str(precursor_family).replace("_", " ")
    if "polysilazane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Polysilazane / polyorganosilazane"]
    elif "carbodiimide" in family_text or "silsesquiazane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Silylcarbodiimide / silylsesquiazane"]
    elif "polycarbosilane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Polycarbosilane"]
    elif "organopolysilane" in family_text or "polysilane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Organopolysilane copolymer"]
    elif "polysilsesquioxane" in family_text or "pms" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Polysilsesquioxane"]
    elif "phtes" in family_text and "teos" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["PhTES + TEOS derived polysiloxane"]
    elif "prtes" in family_text and "teos" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["PrTES + TEOS derived polysiloxane"]
    elif "phtes" in family_text and "mtes" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["PhTES + MTES derived polysiloxane"]
    elif "phtes" in family_text and "vtes" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["PhTES + VTES derived polysiloxane"]
    elif "vtes" in family_text and "mtes" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["VTES + MTES derived polysiloxane"]
    elif "phtes" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["PhTES-derived polysiloxane"]
    elif "vtes" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["VTES-derived polysiloxane"]
    elif "mtes" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["MTES-derived polysiloxane"]
    elif "teos" in family_text or "sol" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["TEOS-derived silica/polysiloxane"]
    elif "alkoxysilane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Alkoxysilane-derived polysiloxane"]
    elif "phenylalkyl polysiloxane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Phenylalkyl polysiloxane"]
    elif "phenyl polysiloxane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Phenyl polysiloxane"]
    elif "vinyl polysiloxane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Vinyl polysiloxane"]
    elif "alkyl polysiloxane" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Alkyl polysiloxane"]
    elif "silicone" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Silicone oil"]
    elif "pitch" in family_text or "carbon" in family_text:
        base = POLYMER_FAMILY_TEMPLATES["Carbon-rich blend"]
    else:
        base = POLYMER_FAMILY_TEMPLATES["Polysiloxane"]
    return ensure_dvb_text(base, bool(dvb_modification), 1.0 if dvb_modification else None)


def suggest_routes(
    target_raw: pd.DataFrame,
    target_engineered: pd.DataFrame,
    engineered_ref: pd.DataFrame,
    n_neighbors: int,
    min_family_match_pct: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if engineered_ref.empty:
        match_table = family_composition_match_table(
            float(target_raw.loc[0, "si_wt_pct"]),
            float(target_raw.loc[0, "c_wt_pct"]),
            float(target_raw.loc[0, "o_wt_pct"]),
            float(target_raw.loc[0, "n_wt_pct"]),
            engineered_ref,
        )
        template_rows = match_table[
            (match_table["label"] != "Other / custom")
            & (match_table["match_pct"] >= min_family_match_pct)
        ].head(max(3, min(n_neighbors, 8)))
        rows = []
        pyro_temp = float(target_raw.loc[0, "pyrolysis_temp_c"])
        pyro_time = float(target_raw.loc[0, "pyrolysis_time_h"])
        for rank, (_, row) in enumerate(template_rows.iterrows(), start=1):
            label = str(row["label"])
            precursor_family = str(row["family"])
            polymer = POLYMER_FAMILY_TEMPLATES.get(label, label)
            recipe = (
                f"Use `{polymer}`, then pyrolyze at {pyro_temp:.0f} C "
                f"for {pyro_time:.2g} h under inert atmosphere. "
                "Adjust the precursor ratio experimentally to reach the entered Si/C/O/N target."
            )
            rows.append({
                "rank": rank,
                "route": f"{precursor_family} | template",
                "suggested_polymer": polymer,
                "recipe": recipe,
                "polymer_family_broad": precursor_family,
                "dvb_modification": "context only",
                "pyrolysis_temp_c": pyro_temp,
                "pyrolysis_time_h": pyro_time,
                "similar_samples": 0,
                "mean_elemental_distance": np.nan,
                "closest_elemental_distance": np.nan,
                "match_pct": float(row["match_pct"]),
                "family_match_pct": float(row["match_pct"]),
                "closest_literature_polymer": "",
                "closest_reference": "",
                "closest_doi": "",
                "source_note": (
                    "Public template route: the private cleaned literature CSV is not loaded, "
                    "so no row-level analog or DOI is shown."
                ),
                "confidence": "template",
                "median_literature_qrev": np.nan,
                "median_literature_qcycled": np.nan,
                "candidate_raw": pd.DataFrame([{
                    "polymer": polymer,
                    "si_wt_pct": float(target_raw.loc[0, "si_wt_pct"]),
                    "c_wt_pct": float(target_raw.loc[0, "c_wt_pct"]),
                    "o_wt_pct": float(target_raw.loc[0, "o_wt_pct"]),
                    "n_wt_pct": float(target_raw.loc[0, "n_wt_pct"]),
                    "pyrolysis_temp_c": pyro_temp,
                    "pyrolysis_time_h": pyro_time,
                }]),
            })
        if not rows:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame(rows), pd.DataFrame()

    neighbors = composition_neighbors(target_engineered, engineered_ref, n_neighbors)
    if neighbors.empty:
        return pd.DataFrame(), neighbors

    match_table = family_composition_match_table(
        float(target_raw.loc[0, "si_wt_pct"]),
        float(target_raw.loc[0, "c_wt_pct"]),
        float(target_raw.loc[0, "o_wt_pct"]),
        float(target_raw.loc[0, "n_wt_pct"]),
        engineered_ref,
    )
    family_match = match_table.set_index("family")["match_pct"].to_dict()
    allowed_families = set(match_table.loc[match_table["match_pct"] >= min_family_match_pct, "family"])
    allowed_families.discard("other")
    if not allowed_families:
        return pd.DataFrame(), neighbors
    filtered_neighbors = neighbors[neighbors["polymer_family_broad"].astype(str).isin(allowed_families)].copy()
    if filtered_neighbors.empty:
        return pd.DataFrame(), neighbors
    neighbors = filtered_neighbors

    group_cols = ["polymer_family_broad", "dvb_modification"]
    rows = []
    numeric_defaults = {
        "pyrolysis_temp_c": safe_series_median(engineered_ref.get("pyrolysis_temp_c", pd.Series(dtype=float)), 1000.0),
        "pyrolysis_time_h": safe_series_median(engineered_ref.get("pyrolysis_time_h", pd.Series(dtype=float)), 1.0),
    }
    for keys, group in neighbors.groupby(group_cols, dropna=False):
        precursor_family, dvb_mod = keys
        dvb_mod = int(0 if pd.isna(dvb_mod) else dvb_mod)
        closest = group.sort_values("elemental_composition_distance").iloc[0]
        closest_polymer = str(closest.get("polymer", representative_polymer(str(precursor_family), dvb_mod)))
        med = {
            "dvb_ratio_to_base": safe_series_median(group.get("dvb_ratio_to_base", pd.Series(dtype=float)), np.nan),
            "pyrolysis_temp_c": safe_series_median(group.get("pyrolysis_temp_c", pd.Series(dtype=float)), numeric_defaults["pyrolysis_temp_c"]),
            "pyrolysis_time_h": safe_series_median(group.get("pyrolysis_time_h", pd.Series(dtype=float)), numeric_defaults["pyrolysis_time_h"]),
        }

        candidate = pd.DataFrame([{
            "polymer": ensure_dvb_text(closest_polymer, bool(dvb_mod), float(med.get("dvb_ratio_to_base", np.nan)) if dvb_mod else None),
            "si_wt_pct": float(target_raw.loc[0, "si_wt_pct"]),
            "c_wt_pct": float(target_raw.loc[0, "c_wt_pct"]),
            "o_wt_pct": float(target_raw.loc[0, "o_wt_pct"]),
            "n_wt_pct": float(target_raw.loc[0, "n_wt_pct"]),
            "pyrolysis_temp_c": float(med.get("pyrolysis_temp_c", numeric_defaults["pyrolysis_temp_c"])),
            "pyrolysis_time_h": float(med.get("pyrolysis_time_h", numeric_defaults["pyrolysis_time_h"])),
            "pyrolysis_atmosphere": "inert",
            "pre_pyrolysis_method": "none",
            "crosslinking_method": "none",
            "crosslink_temp_c": np.nan,
            "crosslink_time_h": np.nan,
            "crosslink_atmosphere": "unknown",
            "first_cycling_current_ma_g": float(target_raw.loc[0, "first_cycling_current_ma_g"]),
            "cycling_numbers": float(target_raw.loc[0, "cycling_numbers"]),
            "cycling_current_ma_g": float(target_raw.loc[0, "cycling_current_ma_g"]),
            "surface_area_m2_g": np.nan,
            "voltage_min_v": 0.0,
            "voltage_max_v": 3.0,
            "voltage_window_v": 3.0,
        }])
        mean_distance = float(group["elemental_composition_distance"].mean())
        closest_distance = float(closest.get("elemental_composition_distance", np.nan))
        match_pct = float(closest.get("composition_match_pct", np.nan))
        sample_support = int(pd.to_numeric(group.get("sample_count", pd.Series([1] * len(group))), errors="coerce").fillna(1).clip(lower=1).sum())
        confidence = "high" if sample_support >= 5 and mean_distance <= 1.5 else "medium" if sample_support >= 2 else "low"
        family_match_pct = float(family_match.get(str(precursor_family), np.nan))
        recipe = (
            f"Use `{candidate.loc[0, 'polymer']}` "
            f"({'with DVB' if dvb_mod else 'without DVB'}), "
            f"then pyrolyze at {float(candidate.loc[0, 'pyrolysis_temp_c']):.0f} C "
            f"for {float(candidate.loc[0, 'pyrolysis_time_h']):.2g} h under inert atmosphere."
        )
        doi_link = normalize_doi(closest.get("doi", ""))
        if bool(closest.get("is_public_aggregate", False)):
            source_note = "Public aggregate route family: no row-level literature entry or DOI is exposed in this deployment."
        else:
            source_note = (
                f"Closest literature guide: [{closest_polymer}]({doi_link})."
                if doi_link
                else "Closest literature guide is listed in the Literature Analogs section."
            )
        rows.append({
            "route": route_label(pd.Series({"polymer_family_broad": precursor_family, "dvb_modification": dvb_mod})),
            "suggested_polymer": candidate.loc[0, "polymer"],
            "recipe": recipe,
            "polymer_family_broad": precursor_family,
            "dvb_modification": "yes" if dvb_mod else "no",
            "pyrolysis_temp_c": float(candidate.loc[0, "pyrolysis_temp_c"]),
            "pyrolysis_time_h": float(candidate.loc[0, "pyrolysis_time_h"]),
            "similar_samples": sample_support,
            "mean_elemental_distance": mean_distance,
            "closest_elemental_distance": closest_distance,
            "match_pct": match_pct,
            "family_match_pct": family_match_pct,
            "closest_literature_polymer": closest_polymer,
            "closest_reference": closest.get("reference", ""),
            "closest_doi": closest.get("doi", ""),
            "source_note": source_note,
            "confidence": confidence,
            "median_literature_qrev": safe_series_median(group.get(TARGET, pd.Series(dtype=float)), np.nan),
            "median_literature_qcycled": safe_series_median(group.get(STABLE_TARGET, pd.Series(dtype=float)), np.nan),
            "candidate_raw": candidate,
        })
    routes = pd.DataFrame(rows).sort_values(
        ["closest_elemental_distance", "mean_elemental_distance", "similar_samples"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    routes.insert(0, "rank", np.arange(1, len(routes) + 1))
    return routes, neighbors


st.markdown(
    """
    <div class="hero">
        <div class="hero-grid">
            <div class="hero-title">
                <h1>SiOC/SiOCN Battery Anode Designer</h1>
                <p>Composition-to-phase screening, synthesis-route suggestions, and leakage-controlled capacity prediction.</p>
            </div>
            <div class="energy-mark" aria-label="battery energy icon">
                <div class="energy-fill"></div>
                <div class="energy-bolt">&#9889;</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

app_target_bundles = load_bundle(APP_TARGET_MODELS_PATH)
discovery_bundle = load_bundle(DISCOVERY_MODEL_PATH)
stable_bundle = load_bundle(STABLE_MODEL_PATH)
stable_diagnostic_bundle = load_bundle(STABLE_DIAGNOSTIC_MODEL_PATH)
irreversible_bundle = load_bundle(IRREVERSIBLE_MODEL_PATH)
ce_bundle = load_bundle(CE_MODEL_PATH)
ce_diagnostic_bundle = load_bundle(CE_DIAGNOSTIC_MODEL_PATH)
reference_raw = load_reference_data(DATA_PATH)
reference_engineered = load_engineered_reference(DATA_PATH)

if not isinstance(app_target_bundles, dict):
    app_target_bundles = {
        "first_reversible": discovery_bundle,
        "ce_design": ce_bundle,
        "ce_diagnostic": ce_diagnostic_bundle,
        "stable_design": stable_bundle,
        "stable_diagnostic": stable_diagnostic_bundle,
        "irreversible_diagnostic_only": irreversible_bundle,
    }

using_public_reference = False
if reference_raw.empty:
    bundled_reference = public_reference_from_bundle(app_target_bundles)
    if not bundled_reference.empty:
        reference_raw = bundled_reference
        reference_engineered = prepare_features(bundled_reference)
        using_public_reference = True

top_left, top_mid, top_right = st.columns([1.2, 1, 1])
with top_left:
    st.markdown("<span class='boxed-label'>Prediction workflow</span>", unsafe_allow_html=True)
    st.caption("First-cycle performance and cyclability are predicted together from one material design.")
with top_mid:
    st.markdown(
        f"<div class='note-card'><b>First-cycle model</b><br>{model_summary(app_target_bundles.get('first_reversible') or discovery_bundle)}</div>",
        unsafe_allow_html=True,
    )
with top_right:
    st.markdown(
        f"<div class='note-card'><b>Cycling diagnostic model</b><br>{model_summary(app_target_bundles.get('stable_diagnostic') or stable_diagnostic_bundle)}</div>",
        unsafe_allow_html=True,
    )

required_bundles = [
    app_target_bundles.get("first_reversible"),
    app_target_bundles.get("ce_design"),
    app_target_bundles.get("ce_diagnostic"),
    app_target_bundles.get("stable_design"),
    app_target_bundles.get("stable_diagnostic"),
]
if any(bundle is None for bundle in required_bundles):
    st.error("The required model bundle is missing. Run the final notebook first.")
    st.stop()

qrev_bundle = app_target_bundles["first_reversible"]
qrev_surface_bundle = app_target_bundles.get("first_reversible_surface")
ce_design_bundle = app_target_bundles["ce_design"]
ce_diag_bundle = app_target_bundles["ce_diagnostic"]
stable_design_bundle = app_target_bundles["stable_design"]
stable_design_surface_bundle = app_target_bundles.get("stable_design_surface")
stable_diag_bundle = app_target_bundles["stable_diagnostic"]
stable_diag_surface_bundle = app_target_bundles.get("stable_diagnostic_surface")
active_bundle = qrev_bundle
selected_target = TARGET
active_features = list(qrev_bundle.get("feature_columns", []))
st.markdown("<span class='boxed-label'>Target material prediction</span>", unsafe_allow_html=True)
st.caption("Enter composition and final pyrolysis conditions first. Synthesis-route suggestions come later and do not change the prediction.")

raw_input, ranges = build_raw_input(reference_raw, reference_engineered, selected_target, active_features)
engineered_input = prepare_features(raw_input)
X_app = feature_matrix(engineered_input.copy(), active_bundle)
prediction = float(active_bundle["model"].predict(X_app)[0])

use_surface_assisted = bool(raw_input.get("use_surface_assisted_ui", pd.Series([False])).iloc[0])
surface_models_available = all(
    bundle is not None
    for bundle in [qrev_surface_bundle, stable_design_surface_bundle, stable_diag_surface_bundle]
)
if use_surface_assisted and surface_models_available:
    qrev_bundle = qrev_surface_bundle
    stable_design_bundle = stable_design_surface_bundle
    stable_diag_bundle = stable_diag_surface_bundle
    active_bundle = qrev_bundle
    X_app = feature_matrix(engineered_input.copy(), active_bundle)
    prediction = float(active_bundle["model"].predict(X_app)[0])
elif use_surface_assisted and not surface_models_available:
    st.warning("Surface-assisted models are not available in the current bundle yet. Using the main no-surface models.")

qrev_pred = max(float(qrev_bundle["model"].predict(feature_matrix(engineered_input.copy(), qrev_bundle))[0]), 0.0)
ce_mode = str(raw_input.get("ce_mode_ui", pd.Series(["Diagnostic CE using predicted Qrev"])).iloc[0])
stable_mode = str(raw_input.get("stable_mode_ui", pd.Series(["Diagnostic Qcycled using predicted Qrev"])).iloc[0])
measured_qrev = pd.to_numeric(raw_input.get("measured_qrev_ui", pd.Series([np.nan])).iloc[0], errors="coerce")
measured_qrev_stable = pd.to_numeric(raw_input.get("measured_qrev_stable_ui", pd.Series([np.nan])).iloc[0], errors="coerce")

if ce_mode == "Design-only CE":
    ce_active_bundle = ce_design_bundle
    qrev_for_ce = qrev_pred
else:
    ce_active_bundle = ce_diag_bundle
    qrev_for_ce = float(measured_qrev) if ce_mode == "Diagnostic CE using measured Qrev" and np.isfinite(measured_qrev) else qrev_pred
ce_engineered = engineered_input.copy()
ce_engineered[TARGET] = qrev_for_ce
ce_pred = float(ce_active_bundle["model"].predict(feature_matrix(ce_engineered.copy(), ce_active_bundle))[0])
ce_pred = float(np.clip(ce_pred, 1.0, 99.9))
qirrev_calc = qrev_for_ce * (100.0 / ce_pred - 1.0)

if stable_mode == "Design-only Qcycled":
    stable_active_bundle = stable_design_bundle
    qrev_for_stable = qrev_pred
else:
    stable_active_bundle = stable_diag_bundle
    qrev_for_stable = (
        float(measured_qrev_stable)
        if stable_mode == "Diagnostic Qcycled using measured Qrev" and np.isfinite(measured_qrev_stable)
        else qrev_pred
    )
stable_engineered = engineered_input.copy()
stable_engineered[TARGET] = qrev_for_stable
cycle_options = [50, 100, 200]
qcycled_raw_by_cycle = {}
for cycle in cycle_options:
    cycle_engineered = stable_engineered.copy()
    cycle_engineered["cycling_numbers"] = cycle
    qcycled_raw_by_cycle[cycle] = max(
        float(stable_active_bundle["model"].predict(feature_matrix(cycle_engineered.copy(), stable_active_bundle))[0]),
        0.0,
    )
qcycled_by_cycle = {}
previous_capacity = qrev_for_stable
for cycle in cycle_options:
    constrained_capacity = min(qcycled_raw_by_cycle[cycle], previous_capacity)
    qcycled_by_cycle[cycle] = constrained_capacity
    previous_capacity = constrained_capacity
selected_cycle = int(engineered_input.loc[0, "cycling_numbers"])
qcycled_pred = qcycled_by_cycle.get(selected_cycle, qcycled_by_cycle[100])
qcycled_was_constrained = any(abs(qcycled_by_cycle[c] - qcycled_raw_by_cycle[c]) > 1e-6 for c in cycle_options)
apparent_retention = 100.0 * qcycled_pred / qrev_for_stable if qrev_for_stable > 0 else np.nan

pred_col, validity_col = st.columns([1.45, 1.0], gap="large")
with pred_col:
    st.subheader("First-cycle performance")
    f1, f2, f3 = st.columns([1.15, 1.15, 0.8])
    with f1:
        prediction_card("Q<sub>Rev</sub>", f"{qrev_pred:.0f}", "mAh/g", "first-cycle reversible", "#236f68")
    with f2:
        prediction_card("Q<sub>Irrev</sub>", f"{qirrev_calc:.0f}", "mAh/g", "calculated first-cycle loss", "#b7791f")
    with f3:
        prediction_card("CE", f"{ce_pred:.1f}", "%", "first-cycle efficiency", "#176b87")
    st.caption(
        "Qirrev is calculated from Qrev and CE: "
        "`Qirrev = Qrev * (100 / CE% - 1)`."
    )
    st.caption(f"CE mode: {ce_mode}.")
    if ce_mode == "Diagnostic CE using measured Qrev":
        st.caption(f"Measured Qrev used for CE/Qirrev calculation: {qrev_for_ce:.0f} mAh/g.")

    st.subheader("Cyclability")
    c1, c2 = st.columns(2)
    cycle_count = int(engineered_input.loc[0, "cycling_numbers"])
    qcycled_note = "design-stage cycled capacity" if stable_mode == "Design-only Qcycled" else "diagnostic cycled capacity"
    with c1:
        prediction_card(
            f"Q<sub>Cycled</sub> at {cycle_count} Cycles",
            f"{qcycled_pred:.1f}",
            "mAh/g",
            qcycled_note,
            "#236f68",
            soft=True,
        )
    with c2:
        prediction_card(
            "Apparent Retention",
            "n/a" if pd.isna(apparent_retention) else f"{apparent_retention:.1f}",
            "" if pd.isna(apparent_retention) else "%",
            "relative to first-cycle Qrev",
            "#176b87",
            soft=True,
        )
    st.caption(f"Cyclability mode: {stable_mode}.")
    if stable_mode == "Diagnostic Qcycled using measured Qrev":
        st.caption(f"Measured Qrev used for cyclability/retention calculation: {qrev_for_stable:.0f} mAh/g.")
    st.caption(
        "Apparent retention is `100 * Qcycled / Qrev`. If first-cycle and cycled capacities were measured at different current densities, "
        "this ratio includes both cycling fade and rate/protocol effects."
    )
    st.caption("Qcycled is capped at Qrev and kept non-increasing with cycle number.")
    if qcycled_was_constrained:
        adjusted_cycles = [
            (cycle, qcycled_raw_by_cycle[cycle], qcycled_by_cycle[cycle])
            for cycle in cycle_options
            if abs(qcycled_raw_by_cycle[cycle] - qcycled_by_cycle[cycle]) > 1e-6
        ]
        st.caption(
            "Cycle trend adjusted: "
            + "; ".join(f"{cycle}-cycle estimate capped from {raw:.1f} to {adj:.1f}" for cycle, raw, adj in adjusted_cycles)
            + " mAh/g."
        )
    st.caption("Cycle-number effects are approximate because the literature data mix different materials rather than full cycling curves for each sample.")
    st.caption("Interpreted at low-current literature conditions, approximately 0.05-0.1C graphite-equivalent and 0-3 V.")
    st.info(graphite_comparison(qrev_pred))
    st.caption(f"Qrev model: {qrev_bundle.get('best_model_name', 'trained model')}")
    st.caption(f"CE model: {ce_active_bundle.get('best_model_name', 'trained model')}")
    st.caption(f"Qcycled model: {stable_active_bundle.get('best_model_name', 'trained model')}")
    st.caption("Polymer, precursor, and DVB are not prediction inputs. They appear only in the synthesis-route suggestions below.")

with validity_col:
    st.subheader("Input range checks")
    for col, label, unit in [
        ("si_wt_pct", "Si", "wt.%"),
        ("c_wt_pct", "C", "wt.%"),
        ("o_wt_pct", "O", "wt.%"),
        ("n_wt_pct", "N", "wt.%"),
        ("pyrolysis_temp_c", "Pyrolysis temperature", "C"),
        ("pyrolysis_time_h", "Pyrolysis time", "h"),
    ]:
        range_caption(label, float(engineered_input.loc[0, col]), ranges[col], unit)
    if use_surface_assisted:
        surface_range = get_numeric_range(reference_engineered, "surface_area_m2_g", (0.0, 100.0, 1500.0))
        range_caption("BET surface area", float(engineered_input.loc[0, "surface_area_m2_g"]), surface_range, "m2/g")
    range_caption("Cycle number", float(engineered_input.loc[0, "cycling_numbers"]), ranges["cycling_numbers"], "cycles")

st.divider()

tab_names = ["Prediction", "Phase Descriptors", "Synthesis Routes", "Literature Analogs", "Model Results", "Tips and Limits"]
tabs = st.tabs(tab_names)
tab_pred = tabs[0]
tab_desc = tabs[1]
tab_route = tabs[2]
tab_analogs = tabs[3]
tab_models = tabs[4]
tab_notes = tabs[5]

with tab_pred:
    c1, c2 = st.columns([1.15, 1])
    with c1:
        plot_prediction_context(reference_engineered, selected_target, prediction)
    with c2:
        st.subheader("Exact model matrix")
        st.dataframe(X_app, width="stretch", hide_index=True)
        st.caption("These are the exact features passed to the trained model.")

with tab_route:
    st.subheader("Literature-guided synthesis routes")
    st.caption(
        "These are recipe ideas for the already-predicted target composition. "
        "They do not change Qrev, CE, Qirrev, or Qcycled above."
    )
    if using_public_reference:
        st.info(
            "Public deployment mode: route suggestions use an aggregate reference library stored in the model bundle. "
            "The private row-level literature CSV is not loaded or exposed."
        )
    elif reference_engineered.empty:
        st.info(
            "Public deployment mode: the private cleaned literature CSV is not loaded. "
            "The app therefore shows template precursor-family routes without row-level analogs, "
            "DOIs, or literature capacities."
        )
    r1, r2 = st.columns([1, 1])
    with r1:
        n_neighbors = st.slider(
            "Number of literature analogs to search",
            min_value=8,
            max_value=40,
            value=20,
            step=2,
            help="How many composition-nearest literature samples are used to build route suggestions. This does not affect the prediction.",
        )
    with r2:
        route_min_match_pct = st.slider(
            "Minimum composition match for route family (%)",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            help="Filters suggested precursor families by elemental Si/C/O/N composition match only. This does not affect the prediction.",
        )
    routes_df, composition_neighbors_df = suggest_routes(
        raw_input,
        engineered_input,
        reference_engineered,
        int(n_neighbors),
        float(route_min_match_pct),
    )
    st.caption(
        "The analog count is the source sample pool. Final route rows can be fewer because samples are grouped by "
        "broad precursor family and DVB context, then filtered by composition match."
    )
    if routes_df.empty:
        st.info("No literature-guided synthesis routes were available for this composition.")
    else:
        if using_public_reference:
            st.caption(
                f"Using {len(composition_neighbors_df)} public aggregate reference route(s), representing "
                f"{int(pd.to_numeric(composition_neighbors_df.get('sample_count', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())} curated literature sample(s), "
                f"grouped into {len(routes_df)} synthesis route suggestion(s)."
            )
        elif reference_engineered.empty:
            st.caption(
                f"Using the public precursor-family template library to show "
                f"{len(routes_df)} synthesis route suggestion(s)."
            )
        else:
            st.caption(
                f"Using {len(composition_neighbors_df)} matched literature sample(s), grouped into "
                f"{len(routes_df)} synthesis route suggestion(s)."
            )
        display_routes = routes_df[[
            c for c in [
                "rank", "suggested_polymer", "recipe", "polymer_family_broad",
                "dvb_modification", "pyrolysis_temp_c", "pyrolysis_time_h",
                "similar_samples", "family_match_pct", "match_pct", "confidence",
                "median_literature_qrev", "median_literature_qcycled",
            ]
            if c in routes_df.columns
        ]].copy()
        st.dataframe(
            display_routes,
            width="stretch",
            hide_index=True,
            column_config={
                "family_match_pct": st.column_config.NumberColumn("family match", format="%.0f%%"),
                "match_pct": st.column_config.NumberColumn("closest composition match", format="%.0f%%"),
                "median_literature_qrev": st.column_config.NumberColumn("median literature Qrev", format="%.0f"),
                "median_literature_qcycled": st.column_config.NumberColumn("median literature Qcycled", format="%.0f"),
            },
        )
        if using_public_reference:
            st.caption(
                "Route matching uses only Si/C/O/N composition distance. Aggregate capacity medians are shown only as context, "
                "not as alternative predictions."
            )
        else:
            st.caption(
                "Route matching uses only Si/C/O/N composition distance. Literature capacities are shown only as context, "
                "not as alternative predictions."
            )
        top_route = routes_df.iloc[0]
        st.subheader("Top recipe idea")
        st.markdown(top_route["recipe"])
        st.markdown(top_route["source_note"])
        if str(top_route["confidence"]) == "template":
            st.caption(
                f"Template composition match: {top_route['match_pct']:.0f}% | "
                "confidence: template route family; no private literature rows are loaded."
            )
        else:
            st.caption(
                f"Closest composition match: {top_route['match_pct']:.0f}% | "
                f"family match: {top_route['family_match_pct']:.0f}% | "
                f"confidence: {top_route['confidence']} from {top_route['similar_samples']} analog sample(s)."
            )

with tab_desc:
    st.subheader("Target composition and process descriptors")
    input_cols = [
        "si_wt_pct", "c_wt_pct", "o_wt_pct", "n_wt_pct",
        "pyrolysis_temp_c", "pyrolysis_time_h", "surface_area_m2_g",
        "cycling_numbers",
    ]
    input_cols = [c for c in input_cols if c in raw_input.columns]
    st.dataframe(raw_input[input_cols], width="stretch", hide_index=True)
    st.caption("These are the user-entered descriptors used by the final prediction workflow. Polymer, precursor, and DVB are reserved for route suggestions.")

    render_phase_dashboard(engineered_input)

    for label, cols in DISPLAY_FEATURE_GROUPS.items():
        present = [c for c in cols if c in engineered_input.columns]
        if present:
            st.subheader(label)
            display_df = engineered_input[present].copy()
            if label == "Composition / phase":
                raw_phase_cols = ["sio2_phase", "sic_phase", "free_c_phase", "si3n4_phase"]
                if all(col in engineered_input.columns for col in raw_phase_cols):
                    phase_total = engineered_input.loc[0, raw_phase_cols].sum()
                    if pd.notna(phase_total) and phase_total > 0:
                        display_df["sio2_normalized_ratio"] = engineered_input.loc[0, "sio2_phase"] / phase_total
                        display_df["sic_normalized_ratio"] = engineered_input.loc[0, "sic_phase"] / phase_total
                        display_df["free_c_normalized_ratio"] = engineered_input.loc[0, "free_c_phase"] / phase_total
                        display_df["si3n4_normalized_ratio"] = engineered_input.loc[0, "si3n4_phase"] / phase_total
                display_df = display_df.rename(columns={
                    "sio2_phase": "sio2_raw_proxy",
                    "sic_phase": "sic_raw_proxy",
                    "free_c_phase": "free_c_raw_proxy",
                    "si3n4_phase": "si3n4_raw_proxy",
                })
            st.dataframe(display_df, width="stretch", hide_index=True)
            if label == "Composition / phase":
                st.caption(
                    "`*_raw_proxy` columns are formula-unit phase proxies from atomic ratios. "
                    "`*_normalized_ratio` columns are the same values divided by their sum and correspond to the percentages above."
                )

with tab_analogs:
    st.subheader("Nearest public aggregate analogs" if using_public_reference else "Nearest literature analogs")
    target_n_for_analogs = float(engineered_input.loc[0, "n_wt_pct"]) if "n_wt_pct" in engineered_input.columns else 0.0
    if target_n_for_analogs >= 1.0:
        st.info("N-containing target detected. Literature analogs and recipe suggestions are restricted to N-containing rows when enough such examples exist.")
    nearest = nearest_literature_rows(raw_input, reference_raw, selected_target)
    if nearest.empty:
        st.info("No comparable literature rows were found.")
    else:
        config = {"doi_link": st.column_config.LinkColumn("DOI / Source")} if "doi_link" in nearest.columns else {}
        st.dataframe(nearest, width="stretch", hide_index=True, column_config=config)
        if using_public_reference:
            st.caption(
                "Analog distance uses only elemental composition: Si, C, O, and N wt.%. "
                "Rows are aggregate public route-family summaries from the private curated dataset; no row-level CSV or DOI is exposed."
            )
        else:
            st.caption("Analog distance uses only elemental composition: Si, C, O, and N wt.%. Process and performance columns are shown only as literature context.")

with tab_models:
    st.subheader("Workflow model bundles")
    bundle_rows = []
    for name, bundle in [
        ("Qrev design model", qrev_bundle),
        ("Qrev surface-assisted model", qrev_surface_bundle),
        ("CE design model", ce_design_bundle),
        ("CE diagnostic model", ce_diag_bundle),
        ("Qcycled design model", stable_design_bundle),
        ("Qcycled surface-assisted model", stable_design_surface_bundle),
        ("Qcycled diagnostic model", stable_diag_bundle),
        ("Qcycled surface-assisted diagnostic model", stable_diag_surface_bundle),
    ]:
        if bundle is None:
            continue
        bundle_rows.append({
            "workflow model": name,
            "target": bundle.get("target"),
            "estimator": bundle.get("best_model_name"),
            "rows": bundle.get("n_training_rows"),
            "groups": bundle.get("n_groups"),
            "features": ", ".join(bundle.get("feature_columns", [])),
        })
    st.dataframe(pd.DataFrame(bundle_rows), width="stretch", hide_index=True)

    st.subheader("Grouped CV vs shuffled diagnostic")
    st.info(
        "Grouped CV keeps all samples from the same DOI/reference in the same fold and is used for model selection. "
        "Shuffled CV randomly splits rows, so related samples from the same paper can appear in both training and test folds. "
        "It is shown only as an optimistic interpolation diagnostic."
    )
    diagnostic_rows = []
    for name, bundle in [
        ("Qrev design", qrev_bundle),
        ("Qrev surface-assisted", qrev_surface_bundle),
        ("CE design", ce_design_bundle),
        ("CE diagnostic", ce_diag_bundle),
        ("Qcycled design", stable_design_bundle),
        ("Qcycled diagnostic", stable_diag_bundle),
        ("Qcycled surface-assisted diagnostic", stable_diag_surface_bundle),
    ]:
        if bundle is None:
            continue
        best = bundle.get("best_by_set")
        feature_set = bundle.get("deployed_feature_set")
        if not isinstance(best, pd.DataFrame) or not feature_set:
            continue
        subset = best[(best["feature_set"] == feature_set) & (best["cv_kind"].isin(["grouped", "shuffled"]))].copy()
        for _, row in subset.iterrows():
            diagnostic_rows.append({
                "workflow model": name,
                "validation": row.get("cv_kind"),
                "feature set": feature_set,
                "estimator": row.get("model"),
                "MAE": row.get("mae_mean"),
                "R2": row.get("test_r2_mean"),
                "rows": row.get("n_rows"),
                "groups": row.get("n_groups"),
            })
    if diagnostic_rows:
        diagnostic_df = pd.DataFrame(diagnostic_rows).sort_values(["workflow model", "validation"])
        st.dataframe(
            diagnostic_df,
            width="stretch",
            hide_index=True,
            column_config={
                "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
                "R2": st.column_config.NumberColumn("R2", format="%.3f"),
            },
        )

    st.subheader("Selected model CV summaries")
    for name, bundle in [
        ("Qrev", qrev_bundle),
        ("CE design", ce_design_bundle),
        ("CE diagnostic", ce_diag_bundle),
        ("Qcycled design/diagnostic comparison", stable_diag_bundle),
    ]:
        best = bundle.get("best_by_set")
        if isinstance(best, pd.DataFrame):
            st.markdown(f"**{name}**")
            st.dataframe(best, width="stretch", hide_index=True)

with tab_notes:
    st.subheader("Practical input tips")
    st.markdown(
        """
        - Enter Si, C, O, and N wt.% plus final pyrolysis temperature/time.
        - The app predicts Qrev and CE, then calculates Qirrev from `Qirrev = Qrev * (100 / CE% - 1)`.
        - Qcycled is predicted at 50, 100, or 200 cycles; apparent retention is `100 * Qcycled / Qrev`.
        - Default CE/Qcycled modes use predicted Qrev as a diagnostic input; measured-Qrev and design-only modes are optional.
        - Phase proxies and formulas are composition-derived screening descriptors, not Raman/NMR/XRD phase quantification.
        - Synthesis routes use polymer/precursor/DVB as literature context only; they do not change the prediction.
        - Predictions are interpreted for low-current 0.05-0.1C graphite-equivalent testing, 0-3 V, and mostly inert/protective pyrolysis.
        - Surface area is optional. Use the surface-assisted model only when BET is measured or intentionally tested.
        - Values near training-range edges should be treated as hypotheses, not guarantees.
        """
    )
    active_summary = pd.DataFrame([
        {"feature group": "Composition", "used by deployed model": "Si, C, O, N wt.%"},
        {"feature group": "Polymer chemistry", "used by deployed model": "Not used directly; route recommendation only"},
        {"feature group": "DVB", "used by deployed model": "Not used directly; route context only"},
        {"feature group": "Pyrolysis", "used by deployed model": "Final pyrolysis temperature and time"},
        {"feature group": "Atmosphere", "used by deployed model": "Fixed interpretation: inert pyrolysis context; diagnostic grouping only in the notebook"},
        {"feature group": "Electrochemical context", "used by deployed model": "Stable target uses cycle number; current density and voltage are fixed low-rate/0-3 V context, not model inputs"},
        {"feature group": "Measured structure", "used by deployed model": "Optional surface-assisted model uses raw BET surface area"},
    ])
    st.dataframe(active_summary, width="stretch", hide_index=True)

st.markdown(
    """
    <div class="app-footer">
        <div>Developed by <b>Dr.-Ing. Maged Bekheet</b></div>
        <div class="footer-links">
            <a class="footer-link" href="https://github.com/magedbekheet" target="_blank" rel="noopener noreferrer">
                <svg class="footer-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5a5.4 5.4 0 0 0-1.5-3.8 5 5 0 0 0-.1-3.2s-1.2-.3-3.9 1.5a13.4 13.4 0 0 0-7 0C4.8 1.2 3.6 1.5 3.6 1.5a5 5 0 0 0-.1 3.2A5.4 5.4 0 0 0 2 8.5c0 3.5 3 5.5 6 5.5-.5.5-.8 1.2-.9 2-.1.8-.1 1.5-.1 2v4"></path>
                    <path d="M9 18c-4.5 2-5-2-7-2"></path>
                </svg>
                GitHub
            </a>
            <a class="footer-link" href="https://www.linkedin.com/in/magedbekheet/" target="_blank" rel="noopener noreferrer">
                <svg class="footer-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6z"></path>
                    <rect x="2" y="9" width="4" height="12"></rect>
                    <circle cx="4" cy="4" r="2"></circle>
                </svg>
                LinkedIn
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
