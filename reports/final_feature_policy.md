# Final Feature Policy for SiOC/SiOCN Anode Discovery

## Goal

Predict and rank polymer-derived Si-O-C and N-doped Si-O-C/Si-C-N anode candidates using descriptors available before electrochemical testing.

The app reports:

- first-cycle reversible capacity (`QRev`)
- first-cycle Coulombic efficiency (`CE`)
- first-cycle irreversible capacity (`QIrrev`), calculated from `QRev` and CE
- cycled reversible capacity (`QCycled`) at 50, 100, or 200 cycles
- apparent retention, calculated as `100 * QCycled / QRev`

## Validation Rule

Use DOI/reference-grouped cross-validation as the main metric. Random shuffled CV is reported only as an optimistic interpolation diagnostic because related samples from the same paper can otherwise appear in both training and test folds.

## Final Deployed First-Cycle Feature Set

The deployed design-stage first-cycle model uses only compact descriptors available before electrochemical testing:

- `si_wt_pct`
- `c_wt_pct`
- `o_wt_pct`
- `n_wt_pct`
- `pyrolysis_temp_c`
- `pyrolysis_time_h`

Current grouped-CV result:

- MAE: `149.45 mAh/g`
- R2: `0.362`
- rows/groups: `219 / 45`

An optional surface-assisted model adds raw `surface_area_m2_g` when BET surface area is known:

- MAE: `146.65 mAh/g`
- R2: `0.389`
- rows/groups: `219 / 45`

## Excluded From the Main Prediction Inputs

- Polymer family and DVB loading are not active capacity-prediction inputs because they made the prediction change even when the final elemental composition stayed fixed. They remain synthesis-route context.
- Current density is not used because the dataset is dominated by low-current measurements and earlier models learned a nonphysical trend.
- Voltage window is not used because most samples are tested near `0-3 V`.
- Pyrolysis atmosphere is not used because nearly all first-capacity rows are interpreted as inert/protective pyrolysis.
- Pre-pyrolysis/crosslinking details are valuable synthesis metadata but too inconsistent for the final deployed model.
- Surface area is optional only because BET reporting is sparse.

The app states these assumptions explicitly and interprets predictions at low-current, `0-3 V` literature conditions.

## Chemistry-Aware Route Guidance

Precursor chemistry is still used for synthesis suggestions, not for capacity prediction. Route families include:

- phenylalkyl, alkyl, vinyl, and phenyl polysiloxanes
- PhTES/MTES/VTES/TEOS-derived polysiloxanes
- polysilazane/polyorganosilazane
- silylcarbodiimide/silylsesquiazane
- polycarbosilane
- polysilsesquioxane
- silicone oil
- carbon-rich blends
- organopolysilane copolymers

Low-N polysiloxane samples are handled as explicit additive routes when nitrogen comes from PVP or pyrrole rather than from the polysiloxane backbone.

The public app can run without the private cleaned CSV. In that mode, it uses conservative template route families and does not show row-level analogs, DOI links, or literature capacities. Local private runs can load `data/sioc_battery_capacity_clean_updated.csv` for full analog search.

## Diagnostic Models

### Coulombic Efficiency

CE can be predicted in two ways:

- design-only CE model using composition + pyrolysis
- diagnostic CE model using predicted or measured first-cycle `QRev`

The app defaults to diagnostic CE using predicted `QRev`, then calculates:

```text
QIrrev = QRev * (100 / CE% - 1)
```

This is preferred over deploying irreversible capacity as an independent card because CE, reversible capacity, and irreversible capacity are directly linked by mass balance.

### Cycled Capacity

The app reports:

- design-only `QCycled` model for stricter pre-test screening
- diagnostic `QCycled` model using predicted or measured first-cycle `QRev`

The default diagnostic `QCycled` model uses:

- `si_wt_pct`
- `c_wt_pct`
- `o_wt_pct`
- `n_wt_pct`
- `pyrolysis_temp_c`
- `pyrolysis_time_h`
- `cycling_numbers`
- predicted or measured first-cycle `QRev`

Current grouped-CV result:

- MAE: `123.09 mAh/g`
- R2: `0.480`
- rows/groups: `123 / 37`

The app constrains `QCycled` predictions so:

```text
Q50 cycles >= Q100 cycles >= Q200 cycles
QCycled <= QRev
```

## Phase-Physics Features

Phase proxies and phase-physics descriptors are useful for discussion and interpretation:

- `sio2_phase`
- `sic_phase`
- `free_c_phase`
- `si3n4_phase`
- `conductive_network_index`
- `ceramic_confinement_index`
- `effective_bandgap_proxy_ev`
- `phase_weighted_electronegativity`
- `phase_separation_index`

They are shown in the app and notebook as composition-derived screening descriptors, not as direct Raman/NMR/XRD phase quantification.

## Features Forbidden for First-Cycle Discovery

Do not use post-electrochemistry descriptors for first-cycle reversible-capacity discovery:

- `reversible_capacity_mah_g`
- `irreversible_capacity_mah_g`
- `coulombic_efficiency_pct`
- `cycling_reversible_capacity_mah_g`
- `capacity_retention_pct`
- measured retention features

These are valid only for explicit diagnostic models where the dependency is stated.

## Final Recommendation

Use `notebooks/05_final_clean_sioc_discovery_modeling.ipynb`, `streamlit_app.py`, and `models/sioc_app_target_models.joblib` as the public project artifacts. Keep the cleaned literature CSV private. The compact composition + pyrolysis model is the primary screening model; phase proxies, SHAP, PCA, and diagnostic CE/cycled-capacity models support interpretation and discussion.
