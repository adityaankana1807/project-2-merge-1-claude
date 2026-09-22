# SANGAM: Big Data-Driven Crime Behaviour Analysis and Case-Linkage Platform

A synthesis of four independently developed prior implementations of the same brief
(project-2-claude, project-2-grok, ihstmo-gemini-project-2,
ihstmoindia-biddata-project-2-codex-1) plus a verified 20-paper literature review,
re-implemented as **SANGAM** ("confluence"): a two-engine platform for India-focused
crime spatio-temporal forecasting and modus-operandi case linkage.

- **Paper**: [`paper/SANGAM_Crime_Linkage_Paper.docx`](paper/SANGAM_Crime_Linkage_Paper.docx)
- **Code**: [`code/ihstmo2/`](code/ihstmo2/) (Python package: `ontology`, `data`, `risk`,
  `cases`, `linkage`, `evaluate`, `lexicon`)
- **Results**: [`code/outputs/full_results.json`](code/outputs/full_results.json)

## What this is

SANGAM-Risk forecasts district-month crime counts using real NCRB district-IPC data
(2001-2012), a genuine queen-contiguity spatial graph built from real district
polygons, causal empirical-Bayes small-area smoothing, and a fitted Negative-Binomial
GLM + gradient-boosted Poisson regressor. SANGAM-Link ranks candidate case pairs for
shared-offender likelihood using adaptive geo-temporal kernels, null-masked
modus-operandi similarity, a multilingual text layer, and an IPC/BNS-aware legal
ontology, evaluated on a purpose-built synthetic FIR benchmark (no real individual-level
Indian case-linkage dataset is publicly available).

See the paper's Section 3 for the comparative audit of the four prior systems that
shaped every design choice here, and Sections 4-6 for full methodology, real results,
and honestly-disclosed limitations (including a residual ceiling effect in the
synthetic linkage benchmark and a numerical-instability finding in NB-GLM fitting that
is documented as a methodological note for future work).

## Reproducing the results

```bash
cd code
pip install -r requirements.txt
python scripts/run_pipeline.py   # writes code/outputs/full_results.json
python scripts/build_paper.py    # regenerates paper/SANGAM_Crime_Linkage_Paper.docx
```

## Data provenance

| File | Source | Real or synthetic |
|---|---|---|
| `data/raw/ncrb_district_ipc_2001_2012.csv` | NCRB district-year IPC panel (public GitHub mirror) | Real |
| `data/raw/census2011_districts.csv` | Census 2011 district demographics | Real |
| `data/geo/india_district.geojson`, `india_district_centroids.csv` | District boundaries (geohacker/india mirror) | Real |
| `data/raw/multilingual_mo_lexicon.csv` | Hand-authored seed lexicon, 6 languages | Synthetic seed, not corpus-derived |
| `data/processed/monthly_district_panel.parquet` | Multinomial disaggregation of real annual totals | Synthetic month placement, real annual totals |
| `data/synthetic/synthetic_cases.parquet` | Generated FIR-level linkage benchmark | Fully synthetic |
