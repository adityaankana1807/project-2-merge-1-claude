"""SANGAM-Link: pairwise case-linkage feature extraction, candidate blocking, and a
calibrated fused ranker for serial-offender case-linkage.

Design principles synthesized from the four-system comparative review and the verified
literature (see paper Sec. 5 for full justification), in priority order matching the
empirical consensus that geo-temporal proximity dominates MO similarity as a linkage
discriminator (Halford 2023: ICD alone AUC=.89 vs MO alone AUC=.58-.66; Borg & Svensson
2022; Tonkin & Woodhams 2017: ICD/TP AUC=.90-.93 vs MO alone .63-.82):
  1. Geo-temporal kernels are the backbone, with adaptive (partially-pooled, ACTUALLY
     fitted, not dead-code) local/global bandwidth shrinkage.
  2. MO similarity is a secondary/refining signal, using null-masking (missing field =
     unknown, never scored as mismatch — independently converged upon by three of the
     four prior systems) plus empirical-Bayes log-odds field weights.
  3. Multilingual free-text similarity (BM25 + lexicon-normalized cosine) is a further
     refining signal, weakest of the three per the literature, weighted accordingly.
  4. A near-repeat/Hawkes-style interaction term and cross-crime legal-family
     compatibility gate (IPC/BNS-aware) complete the feature vector.
  5. Candidate generation uses a real spatial index (BallTree/haversine) rather than an
     O(n) linear scan (a scalability gap flagged in two of the four prior systems).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .lexicon import BM25, MOTokenizer, cosine_bow
from .ontology import get_cross_crime_compatibility

EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# MO similarity: null-masking + empirical-Bayes log-odds field weights
# ---------------------------------------------------------------------------

MO_FIELDS = [
    "entry_method", "weapon_type", "target_type", "property_stolen",
    "transport_mode", "disguise_used", "restraint_used", "time_of_day_band",
]
UNKNOWN = {"UNKNOWN", "", None}


def fit_mo_weights(train_pairs: List[Tuple[dict, dict, bool]]) -> Dict[str, float]:
    """Empirical-Bayes log-odds weight per MO field, fit ONLY on the training split of
    an offender-disjoint pair set: w_k = log[(P(match|linked)+eps)/(P(match|unlinked)+eps)],
    with joint-missingness pairs excluded from both numerator and denominator (the
    null-masking principle applied at fit time too, not just at scoring time)."""
    match_linked = {f: 0 for f in MO_FIELDS}
    obs_linked = {f: 0 for f in MO_FIELDS}
    match_unlinked = {f: 0 for f in MO_FIELDS}
    obs_unlinked = {f: 0 for f in MO_FIELDS}
    for a, b, linked in train_pairs:
        for f in MO_FIELDS:
            va, vb = a.get(f), b.get(f)
            if va in UNKNOWN or vb in UNKNOWN:
                continue
            if linked:
                obs_linked[f] += 1
                match_linked[f] += int(va == vb)
            else:
                obs_unlinked[f] += 1
                match_unlinked[f] += int(va == vb)
    weights = {}
    for f in MO_FIELDS:
        p_link = (match_linked[f] + 1) / (obs_linked[f] + 2)
        p_unlink = (match_unlinked[f] + 1) / (obs_unlinked[f] + 2)
        weights[f] = math.log(p_link / p_unlink)
    return weights


def mo_similarity(a: dict, b: dict, weights: Dict[str, float]) -> Tuple[float, float, List[str]]:
    """Returns (score, coverage, matched_field_explanations). Coverage = fraction of MO
    fields jointly observed; score is the coverage-normalized weighted match sum (0 if
    no field is jointly observed, i.e. treated as fully uninformative rather than as a
    mismatch — this is the concrete answer to FIR-missingness bias)."""
    num, denom, n_observed, explanations = 0.0, 0.0, 0, []
    for f in MO_FIELDS:
        va, vb = a.get(f), b.get(f)
        if va in UNKNOWN or vb in UNKNOWN:
            continue
        n_observed += 1
        w = abs(weights.get(f, 0.0))
        denom += w
        if va == vb:
            num += weights.get(f, 0.0)
            explanations.append(f"{f}={va}")
    coverage = n_observed / len(MO_FIELDS)
    score = (num / denom) if denom > 0 else 0.0
    return score, coverage, explanations


def ifs_fallback_similarity(a: dict, b: dict) -> float:
    """Intuitionistic-fuzzy-set similarity (Dutta & Banik, 2024) for partially-known
    fields: encodes each field as (membership, non-membership, hesitancy) where
    unknown values get high hesitancy rather than being dropped, giving a *softer*
    partial-credit signal than the hard null-masking score above when coverage is very
    low (used only as a fallback feature, not the primary MO score, per Tonkin &
    Woodhams' finding that MO similarity is a comparatively weak standalone predictor)."""
    def ifs(v):
        return (0.84, 0.06, 0.10) if v not in UNKNOWN else (0.22, 0.22, 0.56)

    sims = []
    for f in MO_FIELDS:
        mu_a, nu_a, _ = ifs(a.get(f))
        mu_b, nu_b, _ = ifs(b.get(f))
        if a.get(f) not in UNKNOWN and b.get(f) not in UNKNOWN and a.get(f) != b.get(f):
            mu_b, nu_b = nu_a, mu_a  # disagreement pulls apart
        d_mu, d_nu = abs(mu_a - mu_b), abs(nu_a - nu_b)
        pi_a, pi_b = 1 - mu_a - nu_a, 1 - mu_b - nu_b
        d_pi = abs(pi_a - pi_b)
        sims.append((2 - d_mu - d_nu) / (2 + d_pi))
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Geo-temporal adaptive kernels with real (fitted) partial pooling
# ---------------------------------------------------------------------------

@dataclass
class GeoTemporalScales:
    global_lambda_km: float
    global_tau_days: float
    local_lambda: Dict[str, float] = field(default_factory=dict)
    local_tau: Dict[str, float] = field(default_factory=dict)
    local_n: Dict[str, int] = field(default_factory=dict)
    kappa: float = 15.0

    def lambda_for(self, family: str) -> float:
        n = self.local_n.get(family, 0)
        local = self.local_lambda.get(family, self.global_lambda_km)
        return (n * local + self.kappa * self.global_lambda_km) / (n + self.kappa)

    def tau_for(self, family: str) -> float:
        n = self.local_n.get(family, 0)
        local = self.local_tau.get(family, self.global_tau_days)
        return (n * local + self.kappa * self.global_tau_days) / (n + self.kappa)


def fit_geo_temporal_scales(train_linked_pairs: List[Tuple[float, float, str]]) -> GeoTemporalScales:
    """`train_linked_pairs`: list of (distance_km, day_gap, crime_family) for POSITIVE
    (same-offender) pairs in the training split only. Scales = median distance/day-gap
    among true linked pairs, globally and per crime-family, with the per-family value
    ACTUALLY populated here (the corresponding mechanism was present but never
    populated in one prior system, silently disabling its own shrinkage and degrading
    cross-region transfer)."""
    if not train_linked_pairs:
        return GeoTemporalScales(global_lambda_km=10.0, global_tau_days=30.0)
    dists = [d for d, _, _ in train_linked_pairs]
    gaps = [g for _, g, _ in train_linked_pairs]
    scales = GeoTemporalScales(
        global_lambda_km=float(np.median(dists)) or 10.0,
        global_tau_days=float(np.median(gaps)) or 30.0,
    )
    by_family: Dict[str, List[Tuple[float, float]]] = {}
    for d, g, fam in train_linked_pairs:
        by_family.setdefault(fam, []).append((d, g))
    for fam, vals in by_family.items():
        ds = [v[0] for v in vals]
        gs = [v[1] for v in vals]
        scales.local_lambda[fam] = float(np.median(ds))
        scales.local_tau[fam] = float(np.median(gs))
        scales.local_n[fam] = len(vals)
    return scales


def space_time_kernel(distance_km: float, day_gap: float, family: str,
                       scales: GeoTemporalScales) -> Tuple[float, float]:
    lam = max(scales.lambda_for(family), 0.5)
    tau = max(scales.tau_for(family), 0.5)
    s_space = math.exp(-distance_km / lam)
    s_time = math.exp(-abs(day_gap) / tau)
    return s_space, s_time


# ---------------------------------------------------------------------------
# Full pairwise feature vector
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "s_space", "s_time", "mo_score", "mo_coverage", "ifs_score", "text_score",
    "hawkes_interaction", "crime_compat", "distance_km_capped", "day_gap_capped",
]


def pair_features(case_a: dict, case_b: dict, mo_weights: Dict[str, float],
                   scales: GeoTemporalScales, bm25: BM25, tokenizer: MOTokenizer,
                   compat_table: Optional[Dict] = None) -> Dict[str, float]:
    dist = haversine_km(case_a["lat"], case_a["lon"], case_b["lat"], case_b["lon"])
    gap = abs((case_a["date"] - case_b["date"]).days)
    family = case_a.get("crime_family", "UNKNOWN")
    s_space, s_time = space_time_kernel(dist, gap, family, scales)

    mo_score, coverage, _ = mo_similarity(case_a, case_b, mo_weights)
    ifs_score = ifs_fallback_similarity(case_a, case_b)

    tok_a = tokenizer.tokenize(case_a.get("narrative", ""))
    tok_b = tokenizer.tokenize(case_b.get("narrative", ""))
    text_score = 0.7 * bm25.score_pair_tokens(tok_a, tok_b) / (1 + bm25.score_pair_tokens(tok_a, tok_b)) \
        + 0.3 * cosine_bow(tok_a, tok_b)

    hawkes = s_space * s_time * (1 + max(mo_score, 0) + text_score)
    compat = get_cross_crime_compatibility(
        case_a.get("canonical_class", "THEFT"), case_b.get("canonical_class", "THEFT"),
        learned=compat_table,
    )
    return {
        "s_space": s_space, "s_time": s_time, "mo_score": mo_score,
        "mo_coverage": coverage, "ifs_score": ifs_score, "text_score": text_score,
        "hawkes_interaction": hawkes, "crime_compat": compat,
        "distance_km_capped": min(dist, 300.0) / 300.0,
        "day_gap_capped": min(gap, 365.0) / 365.0,
    }


# ---------------------------------------------------------------------------
# Candidate blocking with a real spatial index
# ---------------------------------------------------------------------------

def build_candidate_index(cases: pd.DataFrame) -> BallTree:
    coords = np.radians(cases[["lat", "lon"]].to_numpy())
    return BallTree(coords, metric="haversine")


def retrieve_candidates(query: pd.Series, cases: pd.DataFrame, tree: BallTree,
                         radius_km: float = 75.0, top_rare_tokens: set | None = None,
                         max_candidates: int = 200) -> List[int]:
    """Union-of-pools blocking: spatial radius (BallTree haversine query) + rare-MO/text
    token inverted-index hits (for cross-region series linked by a distinctive MO term
    rather than proximity) + crime-family compatible pool, deduplicated and capped."""
    q_coord = np.radians([[query["lat"], query["lon"]]])
    radius_rad = radius_km / EARTH_R_KM
    idx_within = tree.query_radius(q_coord, r=radius_rad)[0]
    pool = set(int(i) for i in idx_within if cases.index[i] != query.name and
               cases.iloc[i]["date"] < query["date"])

    if top_rare_tokens:
        q_tokens = set(query.get("mo_tokens", []))
        rare_hit = q_tokens & top_rare_tokens
        if rare_hit:
            mask = cases["mo_tokens"].apply(lambda toks: bool(set(toks) & rare_hit))
            candidate_rows = cases[mask & (cases["date"] < query["date"])]
            pool |= set(cases.index.get_indexer(candidate_rows.index))

    if len(pool) > max_candidates:
        dists = []
        for i in pool:
            d = haversine_km(query["lat"], query["lon"], cases.iloc[i]["lat"], cases.iloc[i]["lon"])
            dists.append((d, i))
        dists.sort()
        pool = set(i for _, i in dists[:max_candidates])
    return sorted(pool)
