"""Real-data harmonization, spatial graph construction, and documented monthly
disaggregation of the real annual NCRB district-IPC panel (2001-2012).

Real data used (see paper Sec. 4 for full provenance):
  - data/raw/ncrb_district_ipc_2001_2012.csv: real NCRB district-year IPC crime counts,
    ~751 districts x 12 years x 29 categories (sourced via the I-HSTMO-Gemini prior
    system's ingestion of a public GitHub mirror of data.gov.in/NCRB).
  - data/raw/census2011_districts.csv: real Census 2011 district demographics.
  - data/geo/india_district.geojson + india_district_centroids.csv: real district
    boundaries/centroids (geohacker/india mirror), used for a genuine queen-contiguity
    adjacency graph rather than an arbitrary k-NN graph (the latter was flagged as a
    simplification in more than one prior system's spatial backbone).

Monthly disaggregation is explicitly SYNTHETIC and documented as such: NCRB does not
publish sub-annual district counts, so every month-level series in this project is a
multinomial reallocation of a real annual total under a documented, non-random
seasonal-weight assumption. The annual totals themselves are always real.
"""
from __future__ import annotations

import difflib
import json
import re
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .config import (CAW_COLUMNS, DATA_GEO, DATA_PROCESSED, DATA_RAW,
                      IPC_COUNT_COLUMNS, NCRB_YEAR_MAX, NCRB_YEAR_MIN, RANDOM_SEED)

STATE_ALIASES = {
    "orissa": "odisha", "pondicherry": "puducherry",
    "uttaranchal": "uttarakhand", "delhi ut": "delhi",
    "andaman & nicobar islands": "andaman and nicobar islands",
    "jammu & kashmir": "jammu and kashmir", "nct of delhi": "delhi",
    "d & n haveli": "dadra and nagar haveli", "d&n haveli": "dadra and nagar haveli",
}


def _normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = STATE_ALIASES.get(s, s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_ncrb_panel() -> pd.DataFrame:
    """Loads the real district-year IPC panel. Drops state-level 'total'/'ut total'
    rows that NCRB's published tables embed as an extra row per state per year inside
    the district list — found by inspection (one such row per state-year, exactly
    matching the number of distinct states) and dropped here, since summing them
    alongside real districts would silently double-count every state's crime total.
    City-police-commissionerate and Government Railway Police (GRP) jurisdiction rows
    (e.g. 'mumbai commr.', 'howrah rly.') are real NCRB-reported units and are KEPT in
    the count panel, but they do not correspond to a single administrative-district
    polygon, so they remain unmatched (isolated) nodes in the spatial graph by
    construction rather than by data-quality error."""
    df = pd.read_csv(DATA_RAW / "ncrb_district_ipc_2001_2012.csv")
    is_total_row = df["district"].str.strip().str.lower().str.fullmatch(r"total|.*ut total")
    df = df[~is_total_row.fillna(False)]
    df["state_key"] = df["state_ut"].map(_normalize)
    df["district_key"] = df["district"].map(_normalize)
    df = df[(df["year"] >= NCRB_YEAR_MIN) & (df["year"] <= NCRB_YEAR_MAX)]
    keep = [c for c in IPC_COUNT_COLUMNS if c in df.columns]
    return df[["state_key", "district_key", "state_ut", "district", "year"] + keep].copy()


def load_census() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "census2011_districts.csv")
    df.columns = [c.strip() for c in df.columns]
    df["state_key"] = df["State name"].map(_normalize)
    df["district_key"] = df["District name"].map(_normalize)
    cols = ["state_key", "district_key", "Population", "Male", "Female", "Literate",
            "Female_Literate", "SC", "ST"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def load_centroids() -> pd.DataFrame:
    df = pd.read_csv(DATA_GEO / "india_district_centroids.csv")
    df["state_key"] = df["state"].map(_normalize)
    df["district_key"] = df["district"].map(_normalize)
    return df


def _fuzzy_match(key: str, candidates_by_state: Dict[str, List[str]], state_key: str,
                  cutoff: float = 0.72) -> Tuple[str | None, float]:
    pool = candidates_by_state.get(state_key, [])
    if not pool:
        return None, 0.0
    match = difflib.get_close_matches(key, pool, n=1, cutoff=cutoff)
    if not match:
        return None, 0.0
    score = difflib.SequenceMatcher(None, key, match[0]).ratio()
    return match[0], score


def harmonize_districts() -> pd.DataFrame:
    """Left-joins NCRB panel with Census 2011 covariates and district centroids on
    normalized (state, district) keys, falling back to per-state fuzzy string matching,
    and finally to state-mean imputation for covariates only (never for the NCRB crime
    counts themselves, which are always kept as observed). Reports match_quality so
    downstream users can filter/weight by data reliability."""
    ncrb = load_ncrb_panel()
    census = load_census()
    centroids = load_centroids()

    districts = ncrb[["state_key", "district_key", "state_ut", "district"]].drop_duplicates()

    def attach(base: pd.DataFrame, other: pd.DataFrame, other_cols: List[str], tag: str) -> pd.DataFrame:
        exact = other.set_index(["state_key", "district_key"])
        pool_by_state: Dict[str, List[str]] = other.groupby("state_key")["district_key"].apply(list).to_dict()
        rows = []
        for _, r in base.iterrows():
            key = (r["state_key"], r["district_key"])
            quality = "none"
            match_row = None
            if key in exact.index:
                match_row = exact.loc[key]
                if isinstance(match_row, pd.DataFrame):
                    match_row = match_row.iloc[0]
                quality = "exact"
            else:
                cand, score = _fuzzy_match(r["district_key"], pool_by_state, r["state_key"])
                if cand is not None:
                    sub = other[(other["state_key"] == r["state_key"]) & (other["district_key"] == cand)]
                    if len(sub):
                        match_row = sub.iloc[0]
                        quality = "fuzzy"
            out = {f"{tag}_match_quality": quality}
            for c in other_cols:
                out[c] = match_row[c] if match_row is not None else np.nan
            rows.append(out)
        attached = pd.DataFrame(rows, index=base.index)
        return pd.concat([base, attached], axis=1)

    census_cols = [c for c in census.columns if c not in ("state_key", "district_key")]
    # Exclude 'district'/'state' name columns from the centroid attach: `districts`
    # already carries the authoritative NCRB district/state_ut names, and attaching a
    # second same-named column here previously produced duplicate-named columns.
    centroid_cols = [c for c in centroids.columns if c not in ("state_key", "district_key", "district", "state")]
    districts = attach(districts, census, census_cols, "census")
    districts = attach(districts, centroids, centroid_cols, "geo")

    # State-mean imputation for unmatched covariates (crime counts are untouched).
    for c in census_cols:
        if pd.api.types.is_numeric_dtype(districts[c]):
            districts[c] = districts.groupby("state_key")[c].transform(lambda s: s.fillna(s.mean()))

    return districts.reset_index(drop=True)


def resolved_coords_dict(harmonized: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Extracts {(state_key, district_key): (centroid_lat, centroid_lon)} from the
    output of `harmonize_districts`, for passing into `build_spatial_graph` so the
    k-NN fallback reuses the already fuzzy-matched coordinates instead of re-deriving
    them from scratch against the raw centroids file."""
    out = {}
    for _, r in harmonized.iterrows():
        key = (r["state_key"], r["district_key"])
        if key not in out and pd.notna(r.get("centroid_lat")) and pd.notna(r.get("centroid_lon")):
            out[key] = (r["centroid_lat"], r["centroid_lon"])
    return out


def build_spatial_graph(district_keys: List[Tuple[str, str]],
                         resolved_coords: Dict[Tuple[str, str], Tuple[float, float]] | None = None,
                         simplify_tol: float = 0.02) -> nx.Graph:
    """Builds a genuine queen-contiguity adjacency graph from the real district
    polygons (shared boundary/vertex touching), falling back to k-NN-by-haversine for
    any (state,district) not matched into the polygon set — an upgrade over a pure k-NN
    graph, which was the spatial backbone used by more than one prior system despite
    India's districts being irregular administrative polygons rather than a uniform
    grid. Geometries are topology-simplified (`simplify_tol` degrees, ~2km) before the
    pairwise buffer/intersects pass, which is a large constant-factor speedup on this
    file's detailed coastline polygons and does not change contiguity results at this
    coarseness. `resolved_coords`, if given (typically the already fuzzy-matched
    centroid columns from `harmonize_districts`), takes priority over a fresh lookup
    against the raw centroids file for the k-NN fallback — using an independent,
    unmatched lookup here was a bug in an earlier version of this function that left
    fuzzy-matched districts as spurious isolates."""
    with open(DATA_GEO / "india_district.geojson", encoding="utf-8") as f:
        gj = json.load(f)

    from shapely.geometry import shape
    from shapely.strtree import STRtree

    geo_key_to_geom = {}
    for feat in gj["features"]:
        props = feat["properties"]
        skey = _normalize(props.get("NAME_1", ""))
        dkey = _normalize(props.get("NAME_2", ""))
        try:
            geom = shape(feat["geometry"]).simplify(simplify_tol, preserve_topology=True)
        except Exception:
            continue
        geo_key_to_geom[(skey, dkey)] = geom

    G = nx.Graph()
    for key in district_keys:
        G.add_node(key)

    matched = {k: geo_key_to_geom[k] for k in district_keys if k in geo_key_to_geom}
    if matched:
        keys_list = list(matched.keys())
        geom_list = [matched[k] for k in keys_list]
        buf_list = [g.buffer(0.01) for g in geom_list]  # ~1km tolerance for shared-boundary touching
        tree = STRtree(buf_list)
        for i, (key_i, buf_i) in enumerate(zip(keys_list, buf_list)):
            for j in tree.query(buf_i):
                key_j = keys_list[j]
                if key_j == key_i or G.has_edge(key_i, key_j):
                    continue
                if buf_i.intersects(buf_list[j]):
                    G.add_edge(key_i, key_j)

    # k-NN fallback (haversine) for unmatched nodes and to guarantee connectivity for
    # isolated polygon matches (e.g. island districts).
    cdict: Dict[Tuple[str, str], Dict[str, float]] = {}
    if resolved_coords:
        for k, (lat, lon) in resolved_coords.items():
            if pd.notna(lat) and pd.notna(lon):
                cdict[k] = {"centroid_lat": lat, "centroid_lon": lon}
    centroids = load_centroids()
    centroids["key"] = list(zip(centroids["state_key"], centroids["district_key"]))
    for _, r in centroids.iterrows():
        cdict.setdefault(r["key"], {"centroid_lat": r["centroid_lat"], "centroid_lon": r["centroid_lon"]})

    def haversine(a, b):
        lat1, lon1, lat2, lon2 = map(np.radians, [a[0], a[1], b[0], b[1]])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * 6371.0088 * np.arcsin(np.sqrt(h))

    keys_with_coords = [k for k in district_keys if k in cdict]
    for key in district_keys:
        if G.degree(key) > 0 or key not in cdict:
            continue
        me = (cdict[key]["centroid_lat"], cdict[key]["centroid_lon"])
        dists = sorted(
            ((haversine(me, (cdict[o]["centroid_lat"], cdict[o]["centroid_lon"])), o)
             for o in keys_with_coords if o != key),
            key=lambda t: t[0],
        )[:6]
        for _, o in dists:
            G.add_edge(key, o)

    return G


_SEASONAL_WEIGHTS = {
    # Deterministic, documented seasonal shape (NOT fit from data): a wedding/festive
    # season bump (Oct-Feb, peaking around Nov) for domestic/dowry-related categories,
    # and a summer-mobility bump (Apr-Jun) for stranger/opportunistic property crime.
    # This is a modelling ASSUMPTION carried over (with the same rationale) from a
    # prior system's synthesize.py, made explicit here rather than hidden.
    "festive": [1.15, 1.05, 0.95, 0.90, 0.90, 0.90, 0.90, 0.90, 0.95, 1.10, 1.30, 1.20],
    "summer_mobility": [0.85, 0.85, 0.95, 1.15, 1.25, 1.15, 0.95, 0.90, 0.90, 0.95, 0.90, 0.90],
    "flat": [1.0] * 12,
}
_CATEGORY_SEASON = {
    "dowry_deaths": "festive",
    "cruelty_by_husband_or_his_relatives": "festive",
    "burglary": "summer_mobility",
    "theft": "summer_mobility",
    "auto_theft": "summer_mobility",
    "dacoity": "summer_mobility",
    "robbery": "summer_mobility",
}


def disaggregate_to_monthly(annual_panel: pd.DataFrame, near_repeat_scale_months: float = 2.5,
                             seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Multinomially reallocates each real district-year-category total into 12 months
    under a documented seasonal weight curve plus a mild within-year near-repeat
    clustering nudge, PRESERVING the annual total exactly (so the real yearly anchor is
    never altered, only its within-year placement is modelled). Returns a long
    district-month-category panel with an `is_synthetic_month` flag."""
    rng = np.random.default_rng(seed)
    rows = []
    cat_cols = [c for c in IPC_COUNT_COLUMNS if c in annual_panel.columns]
    for _, r in annual_panel.iterrows():
        for cat in cat_cols:
            total = r[cat]
            if pd.isna(total) or total <= 0:
                weights = np.array(_SEASONAL_WEIGHTS[_CATEGORY_SEASON.get(cat, "flat")])
                probs = weights / weights.sum()
                counts = np.zeros(12, dtype=int)
            else:
                total = int(round(total))
                season = _CATEGORY_SEASON.get(cat, "flat")
                weights = np.array(_SEASONAL_WEIGHTS[season], dtype=float)
                # small near-repeat nudge: pick one random "seed month" and add a
                # decayed bump, renormalized, so clustering is mild and bounded.
                seed_month = rng.integers(0, 12)
                decay = np.exp(-np.abs(np.arange(12) - seed_month) / near_repeat_scale_months)
                weights = weights * (1.0 + 0.15 * decay)
                probs = weights / weights.sum()
                counts = rng.multinomial(total, probs)
            for m in range(12):
                rows.append({
                    "state_key": r["state_key"], "district_key": r["district_key"],
                    "year": int(r["year"]), "month": m + 1, "category": cat,
                    "count": int(counts[m]), "annual_total": r[cat],
                    "is_synthetic_month": True,
                })
    return pd.DataFrame(rows)


def flag_reporting_shocks(annual_panel: pd.DataFrame, cols: List[str] | None = None,
                           mad_multiplier: float = 3.0) -> pd.DataFrame:
    """Robust year-over-year growth outlier detection per district-category series
    (median + k*MAD threshold), used to flag candidate reporting-infrastructure or
    awareness shocks (e.g. the well-documented 2012-13 national reporting surge in
    crimes-against-women categories following the Nirbhaya case; Mathews et al. 2024)
    rather than to silently 'correct' counts with an unverified multiplier — the latter
    approach appeared in one prior system as a hand-authored per-state constant that
    could not be independently verified and is deliberately NOT reproduced here."""
    cols = cols or CAW_COLUMNS
    df = annual_panel.sort_values(["state_key", "district_key", "year"]).copy()
    flags = []
    for (s, d), g in df.groupby(["state_key", "district_key"]):
        g = g.sort_values("year")
        for col in cols:
            if col not in g.columns:
                continue
            vals = g[col].to_numpy(dtype=float)
            if len(vals) < 3:
                continue
            growth = np.diff(vals) / np.maximum(vals[:-1], 1.0)
            med = np.median(growth)
            mad = np.median(np.abs(growth - med)) + 1e-6
            outlier_years = g["year"].to_numpy()[1:][np.abs(growth - med) > mad_multiplier * mad]
            for y in outlier_years:
                flags.append({"state_key": s, "district_key": d, "category": col, "year": int(y)})
    return pd.DataFrame(flags)
