"""Synthetic FIR-level case-linkage benchmark generator.

No real individual-level (offender/FIR) Indian crime-linkage dataset is publicly
available (all four prior systems independently confirmed this and built synthetic
benchmarks for the same reason). This generator combines the strongest elements found
across those four attempts plus the literature review, and is explicit that it
produces SYNTHETIC ground truth: reported linkage metrics measure whether the pipeline
recovers a KNOWN synthetic generative process, not real-world case-linkage accuracy.

Design choices and their justification:
  - Locations are sampled from the REAL harmonized district set (population-weighted),
    not a handful of hardcoded cities (broader geographic coverage than one prior
    system's 4-state benchmark).
  - Three offender spatial typologies (forager / marauder / commuter), following
    Halford (2023)'s optimal-forager-patch framing and one prior system's typology
    mixture idea, but as an explicit GENERATIVE assumption used to build ground truth
    (not smuggled in as a fitted model claim).
  - MO fields drawn from crime-type-conditional pools with a tunable per-offender
    consistency rate and a forced-UNKNOWN masking rate, so the null-masking scorer in
    linkage.py is genuinely exercised (not just a defensive no-op).
  - A `held_out` flag independently drops a fraction of each series' cases from the
    observed dataset (never removing the ground-truth label used for evaluation) to
    stress-test linkage robustness to case attrition/underreporting, WITHOUT claiming
    the drop rate is a validated real-world underreporting statistic.
  - IPC/BNS legal regime assigned by the real 1 July 2024 transition date, spreading
    case dates 2018-2025 so both regimes are represented and a temporal holdout across
    the transition is possible.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import RANDOM_SEED
from .ontology import CANONICAL_CLASSES, CLASS_TO_FAMILY, get_cross_crime_compatibility

BNS_TRANSITION = dt.date(2024, 7, 1)

MO_POOLS: Dict[str, Dict[str, List[str]]] = {
    "BURGLARY_HOUSEBREAKING": {
        "entry_method": ["window_grill_cut", "door_lock_broken", "roof_access", "wall_breach"],
        "weapon_type": ["UNKNOWN", "crowbar", "knife"],
        "target_type": ["residential", "commercial"],
        "property_stolen": ["jewellery", "cash", "electronics"],
        "transport_mode": ["motorcycle", "on_foot", "UNKNOWN"],
        "disguise_used": ["mask", "UNKNOWN"],
        "restraint_used": ["UNKNOWN", "rope"],
        "time_of_day_band": ["night", "late_night", "early_morning"],
    },
    "THEFT": {
        "entry_method": ["UNKNOWN", "opportunistic"],
        "weapon_type": ["UNKNOWN"],
        "target_type": ["street", "market", "residential"],
        "property_stolen": ["cash", "mobile_phone", "chain_snatching"],
        "transport_mode": ["motorcycle", "on_foot"],
        "disguise_used": ["UNKNOWN", "helmet"],
        "restraint_used": ["UNKNOWN"],
        "time_of_day_band": ["day", "evening"],
    },
    "VEHICLE_THEFT": {
        "entry_method": ["hotwire", "duplicate_key", "lock_broken"],
        "weapon_type": ["UNKNOWN"],
        "target_type": ["parked_vehicle"],
        "property_stolen": ["motorcycle", "car"],
        "transport_mode": ["UNKNOWN"],
        "disguise_used": ["UNKNOWN"],
        "restraint_used": ["UNKNOWN"],
        "time_of_day_band": ["night", "day"],
    },
    "ROBBERY": {
        "entry_method": ["UNKNOWN", "street_confrontation"],
        "weapon_type": ["knife", "firearm", "UNKNOWN"],
        "target_type": ["individual", "shop"],
        "property_stolen": ["cash", "jewellery", "mobile_phone"],
        "transport_mode": ["motorcycle", "on_foot"],
        "disguise_used": ["mask", "UNKNOWN"],
        "restraint_used": ["UNKNOWN"],
        "time_of_day_band": ["evening", "night"],
    },
    "DACOITY": {
        "entry_method": ["forced_entry", "door_broken"],
        "weapon_type": ["firearm", "knife"],
        "target_type": ["residential", "commercial"],
        "property_stolen": ["cash", "jewellery"],
        "transport_mode": ["car", "motorcycle"],
        "disguise_used": ["mask"],
        "restraint_used": ["rope", "UNKNOWN"],
        "time_of_day_band": ["night"],
    },
    "EXTORTION": {
        "entry_method": ["UNKNOWN"],
        "weapon_type": ["UNKNOWN", "firearm"],
        "target_type": ["shop_owner", "individual"],
        "property_stolen": ["cash"],
        "transport_mode": ["UNKNOWN"],
        "disguise_used": ["UNKNOWN"],
        "restraint_used": ["UNKNOWN"],
        "time_of_day_band": ["day", "evening"],
    },
    "SEXUAL_OFFENCE": {
        "entry_method": ["UNKNOWN"],
        "weapon_type": ["UNKNOWN", "threat"],
        "target_type": ["individual"],
        "property_stolen": ["UNKNOWN"],
        "transport_mode": ["UNKNOWN", "motorcycle"],
        "disguise_used": ["UNKNOWN"],
        "restraint_used": ["UNKNOWN", "rope"],
        "time_of_day_band": ["night", "evening", "day"],
    },
    "HOMICIDE": {
        "entry_method": ["UNKNOWN"],
        "weapon_type": ["knife", "firearm", "blunt_object"],
        "target_type": ["individual"],
        "property_stolen": ["UNKNOWN"],
        "transport_mode": ["UNKNOWN"],
        "disguise_used": ["UNKNOWN"],
        "restraint_used": ["UNKNOWN"],
        "time_of_day_band": ["night", "day"],
    },
    "KIDNAPPING_ABDUCTION": {
        "entry_method": ["UNKNOWN"],
        "weapon_type": ["UNKNOWN", "threat"],
        "target_type": ["individual"],
        "property_stolen": ["UNKNOWN"],
        "transport_mode": ["car", "motorcycle"],
        "disguise_used": ["UNKNOWN"],
        "restraint_used": ["rope", "UNKNOWN"],
        "time_of_day_band": ["day", "evening"],
    },
    "CYBER_FINANCIAL": {
        "entry_method": ["phishing_link", "otp_fraud", "fake_call"],
        "weapon_type": ["UNKNOWN"],
        "target_type": ["individual", "bank_account"],
        "property_stolen": ["cash"],
        "transport_mode": ["UNKNOWN"],
        "disguise_used": ["UNKNOWN"],
        "restraint_used": ["UNKNOWN"],
        "time_of_day_band": ["UNKNOWN"],
    },
}

NARRATIVE_TEMPLATES = [
    "Complainant reported that unknown accused entered via {entry_method} and took away {property_stolen}. {vernacular}",
    "FIR states offender(s) used {weapon_type} near {target_type}; escaped on {transport_mode}. {vernacular}",
    "Accused, wearing {disguise_used}, committed the offence at {time_of_day_band}; {vernacular}",
]

TYPOLOGY_PARAMS = {
    # Generative spatial dispersion (approx std-dev of series-member offset from a
    # single home location, in km) — an explicit modelling assumption, not a fitted
    # real-world offender-mobility statistic.
    "marauder": {"jitter_km": 8.0, "weight": 0.5},
    "forager": {"jitter_km": 25.0, "weight": 0.3},
    "commuter": {"jitter_km": 60.0, "weight": 0.2},
}


def _offset_latlon(lat, lon, km, rng):
    bearing = rng.uniform(0, 2 * math.pi)
    dlat = (km / 111.0) * math.cos(bearing)
    dlon = (km / (111.0 * max(math.cos(math.radians(lat)), 0.1))) * math.sin(bearing)
    return lat + dlat, lon + dlon


@dataclass
class CaseGenConfig:
    n_offenders_target_singletons: int = 700
    n_offenders_multi: int = 300
    n_districts_pool: int = 300
    consistency_rate: float = 0.78
    unknown_mask_rate: float = 0.18
    versatile_rate: float = 0.22
    held_out_rate: float = 0.12
    date_start: dt.date = dt.date(2018, 1, 1)
    date_end: dt.date = dt.date(2025, 12, 31)
    seed: int = RANDOM_SEED


def generate_synthetic_cases(district_pool: pd.DataFrame, cfg: CaseGenConfig = CaseGenConfig()) -> pd.DataFrame:
    """`district_pool` must have columns [state_key, district_key, lat, lon, population]
    (population may contain NaN; treated as uniform weight in that case)."""
    rng = np.random.default_rng(cfg.seed)
    pool = district_pool.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(pool) > cfg.n_districts_pool:
        w = pool["population"].fillna(pool["population"].median()).to_numpy()
        w = w / w.sum()
        idx = rng.choice(len(pool), size=cfg.n_districts_pool, replace=False, p=w)
        pool = pool.iloc[idx].reset_index(drop=True)

    typologies = list(TYPOLOGY_PARAMS.keys())
    typology_p = [TYPOLOGY_PARAMS[t]["weight"] for t in typologies]

    n_days = (cfg.date_end - cfg.date_start).days
    cases = []
    case_id = 0
    n_offenders_total = cfg.n_offenders_target_singletons + cfg.n_offenders_multi

    for off_i in range(n_offenders_total):
        is_singleton = off_i < cfg.n_offenders_target_singletons
        home_row = pool.iloc[rng.integers(0, len(pool))]
        home_lat, home_lon = home_row["lat"], home_row["lon"]
        typology = rng.choice(typologies, p=typology_p)
        primary_class = rng.choice(CANONICAL_CLASSES)
        family = CLASS_TO_FAMILY[primary_class]
        is_versatile = (not is_singleton) and (rng.random() < cfg.versatile_rate)
        secondary_class = None
        if is_versatile:
            same_family = [c for c in CANONICAL_CLASSES if CLASS_TO_FAMILY[c] == family and c != primary_class]
            secondary_class = rng.choice(same_family) if same_family else None

        if is_singleton:
            series_len = 1
        else:
            r = rng.random()
            series_len = 2 if r < 0.55 else (rng.integers(3, 6) if r < 0.85 else rng.integers(6, 10))

        # consistent MO profile for this offender, generated once
        base_mo = {f: rng.choice(MO_POOLS[primary_class][f]) for f in MO_POOLS[primary_class]}
        start_day = rng.integers(0, max(n_days - 30 * series_len, 1))

        for s in range(int(series_len)):
            cls = primary_class
            if secondary_class is not None and s % 2 == 1:
                cls = secondary_class
            jitter = TYPOLOGY_PARAMS[typology]["jitter_km"]
            lat, lon = _offset_latlon(home_lat, home_lon, rng.exponential(jitter), rng)
            gap_days = int(rng.exponential(18)) if s > 0 else 0
            case_date = cfg.date_start + dt.timedelta(days=int(start_day) + sum(
                int(rng.exponential(18)) for _ in range(s)
            ))
            if case_date > cfg.date_end:
                case_date = cfg.date_end

            mo = {}
            for f, pool_vals in MO_POOLS[cls].items():
                if rng.random() < cfg.unknown_mask_rate:
                    mo[f] = "UNKNOWN"
                elif f in base_mo and cls == primary_class and rng.random() < cfg.consistency_rate:
                    mo[f] = base_mo[f]
                else:
                    mo[f] = rng.choice(pool_vals)

            regime = "BNS" if case_date >= BNS_TRANSITION else "IPC"
            narrative = rng.choice(NARRATIVE_TEMPLATES).format(
                entry_method=mo.get("entry_method", "UNKNOWN"),
                property_stolen=mo.get("property_stolen", "UNKNOWN"),
                weapon_type=mo.get("weapon_type", "UNKNOWN"),
                target_type=mo.get("target_type", "UNKNOWN"),
                transport_mode=mo.get("transport_mode", "UNKNOWN"),
                disguise_used=mo.get("disguise_used", "UNKNOWN"),
                time_of_day_band=mo.get("time_of_day_band", "UNKNOWN"),
                vernacular=rng.choice(["taala tod kar ghusa", "gala ghont kar", "desi katta dikhaya", ""]),
            )

            held_out = rng.random() < cfg.held_out_rate
            cases.append({
                "case_id": case_id, "offender_id": off_i, "typology": typology,
                "canonical_class": cls, "crime_family": CLASS_TO_FAMILY[cls],
                "legal_regime": regime, "lat": lat, "lon": lon,
                "state_key": home_row["state_key"], "district_key": home_row["district_key"],
                "date": pd.Timestamp(case_date), "narrative": narrative,
                "is_singleton_offender": is_singleton, "is_versatile": is_versatile,
                "held_out": held_out,
                **mo,
            })
            case_id += 1

    df = pd.DataFrame(cases)
    df["mo_tokens"] = df["narrative"].apply(lambda s: [])  # filled in later with the real tokenizer
    return df


def offender_disjoint_split(cases: pd.DataFrame, train_frac=0.7, val_frac=0.15,
                             seed: int = RANDOM_SEED):
    offenders = cases["offender_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(offenders)
    n = len(offenders)
    train_off = set(offenders[: int(n * train_frac)])
    val_off = set(offenders[int(n * train_frac): int(n * (train_frac + val_frac))])
    test_off = set(offenders[int(n * (train_frac + val_frac)):])
    return (
        cases[cases["offender_id"].isin(train_off)].copy(),
        cases[cases["offender_id"].isin(val_off)].copy(),
        cases[cases["offender_id"].isin(test_off)].copy(),
    )


def leave_state_out_split(cases: pd.DataFrame, test_states: List[str]):
    test_mask = cases["state_key"].isin(test_states)
    return cases[~test_mask].copy(), cases[test_mask].copy()


def temporal_bns_split(cases: pd.DataFrame):
    pre = cases[cases["date"] < pd.Timestamp(BNS_TRANSITION)].copy()
    post = cases[cases["date"] >= pd.Timestamp(BNS_TRANSITION)].copy()
    return pre, post
