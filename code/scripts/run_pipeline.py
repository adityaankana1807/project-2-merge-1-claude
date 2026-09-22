"""End-to-end SANGAM pipeline: harmonize real data -> build spatial graph -> regional
risk forecasting (SAE + NB-GLM + HGB-Poisson) -> space-time scan -> synthetic case
generation -> case-linkage feature fitting -> evaluation -> outputs/*.json.

Run from the `code/` directory: `python scripts/run_pipeline.py`
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ihstmo2 import cases as cases_mod
from ihstmo2 import data, evaluate, linkage, ontology, risk
from ihstmo2.config import DATA_SYNTHETIC, OUTPUTS_DIR
from ihstmo2.lexicon import BM25, MOTokenizer


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    out = {}

    # ---------------- 1. Real data harmonization ----------------
    log("Harmonizing NCRB district-IPC panel with Census 2011 + district geometry...")
    harmonized = data.harmonize_districts()
    out["n_districts"] = int(harmonized[["state_key", "district_key"]].drop_duplicates().shape[0])
    out["census_match_quality"] = harmonized["census_match_quality"].value_counts().to_dict()
    out["geo_match_quality"] = harmonized["geo_match_quality"].value_counts().to_dict()

    keys = list(dict.fromkeys(zip(harmonized["state_key"], harmonized["district_key"])))
    coords = data.resolved_coords_dict(harmonized)
    log(f"Building spatial graph over {len(keys)} districts...")
    graph = data.build_spatial_graph(keys, resolved_coords=coords)
    out["graph_nodes"] = graph.number_of_nodes()
    out["graph_edges"] = graph.number_of_edges()
    out["graph_isolates"] = len(list(__import__("networkx").isolates(graph)))

    # ---------------- 2. Reporting-shock flagging ----------------
    ncrb_panel = data.load_ncrb_panel()
    log("Flagging candidate reporting-infrastructure shocks in CAW categories...")
    shocks = data.flag_reporting_shocks(ncrb_panel)
    out["reporting_shock_flags_n"] = int(len(shocks))

    # ---------------- 3. Monthly disaggregation ----------------
    log("Disaggregating real annual totals to monthly (documented seasonal model)...")
    monthly = data.disaggregate_to_monthly(ncrb_panel)
    monthly.to_parquet(data.DATA_PROCESSED / "monthly_district_panel.parquet", index=False)
    out["monthly_panel_rows"] = int(len(monthly))

    # ---------------- 4. Forecasting: theft (high-volume, spatially structured) ----------------
    forecast_results = {}
    for category in ["theft", "burglary", "dowry_deaths"]:
        log(f"Building forecast features + fitting models for category={category}...")
        feats = risk.build_forecast_features(monthly, graph, category=category, max_hop=2)
        if len(feats) < 200:
            log(f"  skipping {category}: insufficient rows ({len(feats)})")
            continue
        res, _ = risk.fit_and_evaluate_forecast(feats)
        forecast_results[category] = res
        log(f"  {category}: {json.dumps(res, default=float)[:300]}")
    out["forecast_results"] = forecast_results

    # ---------------- 5. Synthetic case-linkage benchmark ----------------
    log("Generating synthetic FIR-level case-linkage benchmark...")
    district_pool = harmonized.rename(columns={"centroid_lat": "lat", "centroid_lon": "lon",
                                                 "Population": "population"})
    district_pool = district_pool.dropna(subset=["lat", "lon"])
    syn_cases = cases_mod.generate_synthetic_cases(district_pool)
    out["n_synthetic_cases"] = int(len(syn_cases))
    out["n_synthetic_offenders"] = int(syn_cases["offender_id"].nunique())
    syn_cases.drop(columns=["mo_tokens"]).to_parquet(DATA_SYNTHETIC / "synthetic_cases.parquet", index=False)

    observed = syn_cases[~syn_cases["held_out"]].reset_index(drop=True)
    train_c, val_c, test_c = cases_mod.offender_disjoint_split(observed)
    log(f"Offender-disjoint split: train={len(train_c)} val={len(val_c)} test={len(test_c)}")

    tokenizer = MOTokenizer()
    all_tokens = [tokenizer.tokenize(t) for t in observed["narrative"]]
    bm25 = BM25(all_tokens)

    def make_pairs(df, max_pairs_per_query=40, seed=0, hard_negative_radius_km=40.0,
                    hard_negative_frac=0.8):
        """Positive pairs = all earlier same-offender cases. Negative pairs are drawn
        MOSTLY from the query's own tight geographic candidate pool (BallTree radius
        search within `hard_negative_radius_km`, preferring same-crime-family
        candidates) rather than uniformly from the whole country. Uniform-random
        negatives are trivially separable by raw distance alone (most of India is
        hundreds of km from any given case), which initially produced a meaningless
        AUC around 0.9998 — the exact ceiling-effect failure mode flagged in the
        review of one of the four prior systems. A 150km radius still left the task
        too easy (AUC 0.999); 40km approximately matches this generator's own
        marauder/forager typology jitter scale (8-25km), so same- and different-
        offender cases genuinely overlap in space, and restricting to the same crime
        family removes the trivial cross-family `crime_compat` shortcut, forcing the
        MO/text features to do real work. Mixing in a `1 - hard_negative_frac` tail of
        genuinely random negatives keeps some easy-negative calibration signal without
        letting it dominate the metric."""
        rng = np.random.default_rng(seed)
        pairs = []
        by_offender = df.groupby("offender_id").indices
        idx_list = df.index.to_numpy()
        tree = linkage.build_candidate_index(df)
        n_hard = max(1, int(round(max_pairs_per_query * hard_negative_frac)))
        n_easy = max(0, max_pairs_per_query - n_hard)
        for i in idx_list:
            row = df.loc[i]
            same = [df.index[j] for j in by_offender.get(row["offender_id"], []) if df.index[j] != i]
            for j_idx in same:
                if df.loc[j_idx, "date"] < row["date"]:
                    pairs.append((i, j_idx, True))

            candidate_pos = linkage.retrieve_candidates(row, df, tree, radius_km=hard_negative_radius_km,
                                                          max_candidates=n_hard * 6)
            candidate_idx = [df.index[p] for p in candidate_pos
                              if df.index[p] != i and df.loc[df.index[p], "offender_id"] != row["offender_id"]]
            same_family = [j for j in candidate_idx if df.loc[j, "crime_family"] == row["crime_family"]]
            other_family = [j for j in candidate_idx if j not in set(same_family)]
            rng.shuffle(same_family)
            rng.shuffle(other_family)
            candidate_idx = same_family + other_family
            added = set()
            for j_idx in candidate_idx[:n_hard]:
                pairs.append((i, j_idx, False))
                added.add(j_idx)

            if n_easy > 0:
                easy_pool = rng.choice(idx_list, size=min(n_easy * 3, len(idx_list) - 1), replace=False)
                n_added = 0
                for j_idx in easy_pool:
                    if j_idx == i or j_idx in added or df.loc[j_idx, "offender_id"] == row["offender_id"]:
                        continue
                    if df.loc[j_idx, "date"] >= row["date"]:
                        continue
                    pairs.append((i, j_idx, False))
                    n_added += 1
                    if n_added >= n_easy:
                        break
        return pairs

    log("Building training pairs and fitting MO weights + geo-temporal scales...")
    train_pair_idx = make_pairs(train_c, max_pairs_per_query=20)
    train_pairs_dicts = [(train_c.loc[i].to_dict(), train_c.loc[j].to_dict(), lab) for i, j, lab in train_pair_idx]
    mo_weights = linkage.fit_mo_weights(train_pairs_dicts)
    out["mo_field_weights"] = mo_weights

    linked_geo = [
        (linkage.haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]),
         abs((a["date"] - b["date"]).days), a.get("crime_family", "UNKNOWN"))
        for a, b, lab in train_pairs_dicts if lab
    ]
    scales = linkage.fit_geo_temporal_scales(linked_geo)
    out["geo_temporal_scales"] = {
        "global_lambda_km": scales.global_lambda_km, "global_tau_days": scales.global_tau_days,
        "local_lambda": scales.local_lambda, "local_tau": scales.local_tau,
    }

    compat_pairs = [(a.get("canonical_class"), b.get("canonical_class"), lab) for a, b, lab in train_pairs_dicts]
    learned_compat = ontology.calibrate_compatibility_from_pairs(compat_pairs)

    def featurize(pair_idx_list, df):
        rows = []
        for i, j, lab in pair_idx_list:
            a, b = df.loc[i].to_dict(), df.loc[j].to_dict()
            f = linkage.pair_features(a, b, mo_weights, scales, bm25, tokenizer, compat_table=learned_compat)
            f["label"] = lab
            f["query_id"] = i
            rows.append(f)
        return pd.DataFrame(rows)

    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    train_feat = featurize(train_pair_idx, train_c)
    log(f"Fitting calibrated logistic ranker on {len(train_feat)} training pairs...")
    X_train = train_feat[linkage.FEATURE_NAMES].to_numpy()
    y_train = train_feat["label"].to_numpy().astype(int)
    base = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    clf.fit(X_train, y_train)

    coefs = dict(zip(linkage.FEATURE_NAMES, base.fit(X_train, y_train).coef_[0].tolist()))
    out["linkage_feature_coefficients"] = coefs

    def evaluate_split(df, name):
        pair_idx_list = make_pairs(df, max_pairs_per_query=60, seed=42)
        feat = featurize(pair_idx_list, df)
        if len(feat) == 0:
            return {}
        X = feat[linkage.FEATURE_NAMES].to_numpy()
        proba = clf.predict_proba(X)[:, 1]
        feat["score"] = proba
        pw = evaluate.pairwise_metrics(feat["label"].to_numpy(), proba)
        query_dfs = [g.rename(columns={"label": "is_match"}) for _, g in feat.groupby("query_id")]
        rk = evaluate.ranking_metrics(query_dfs)
        return {**pw, **rk}

    log("Evaluating on held-out val/test splits...")
    val_metrics = evaluate_split(val_c, "val")
    test_metrics = evaluate_split(test_c, "test")
    out["linkage_val_metrics"] = val_metrics
    out["linkage_test_metrics"] = test_metrics

    log("Running ablations (single-feature-group rankers)...")
    ablation_groups = {
        "space_only": ["s_space", "distance_km_capped"],
        "time_only": ["s_time", "day_gap_capped"],
        "mo_only": ["mo_score", "mo_coverage", "ifs_score"],
        "text_only": ["text_score"],
        "geo_temporal": ["s_space", "s_time", "distance_km_capped", "day_gap_capped"],
        "full": linkage.FEATURE_NAMES,
    }
    test_pair_idx = make_pairs(test_c, max_pairs_per_query=60, seed=42)
    test_feat_full = featurize(test_pair_idx, test_c)
    ablation_rows = []
    for name, feats_subset in ablation_groups.items():
        Xa_train = train_feat[feats_subset].to_numpy()
        ya_train = train_feat["label"].to_numpy().astype(int)
        m = LogisticRegression(class_weight="balanced", max_iter=1000)
        m.fit(Xa_train, ya_train)
        Xa_test = test_feat_full[feats_subset].to_numpy()
        proba = m.predict_proba(Xa_test)[:, 1]
        pw = evaluate.pairwise_metrics(test_feat_full["label"].to_numpy(), proba)
        qd = test_feat_full.assign(score=proba).rename(columns={"label": "is_match"})
        rk = evaluate.ranking_metrics([g for _, g in qd.groupby("query_id")])
        ablation_rows.append({"feature_set": name, **pw, **rk})
    out["linkage_ablations"] = ablation_rows

    log("Cross-state (leave-state-out) generalization test...")
    all_states = observed["state_key"].unique().tolist()
    rng = np.random.default_rng(1)
    test_states = list(rng.choice(all_states, size=max(1, len(all_states) // 5), replace=False))
    train_geo, test_geo = cases_mod.leave_state_out_split(observed, test_states)
    train_geo_off, _, _ = cases_mod.offender_disjoint_split(train_geo)
    tp_idx = make_pairs(train_geo_off, max_pairs_per_query=20)
    tp_dicts = [(train_geo_off.loc[i].to_dict(), train_geo_off.loc[j].to_dict(), l) for i, j, l in tp_idx]
    mo_w2 = linkage.fit_mo_weights(tp_dicts)
    geo2 = [(linkage.haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]), abs((a["date"] - b["date"]).days),
             a.get("crime_family", "UNKNOWN")) for a, b, l in tp_dicts if l]
    scales2 = linkage.fit_geo_temporal_scales(geo2)
    tf2 = pd.DataFrame([
        {**linkage.pair_features(a, b, mo_w2, scales2, bm25, tokenizer), "label": l, "query_id": i}
        for (i, j, l), (a, b, _) in zip(tp_idx, tp_dicts)
    ])
    m2 = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
        tf2[linkage.FEATURE_NAMES], tf2["label"].astype(int))
    test_geo_idx = make_pairs(test_geo, max_pairs_per_query=60, seed=7)
    test_geo_feat = featurize(test_geo_idx, test_geo)
    if len(test_geo_feat):
        proba_geo = m2.predict_proba(test_geo_feat[linkage.FEATURE_NAMES])[:, 1]
        pw_geo = evaluate.pairwise_metrics(test_geo_feat["label"].to_numpy(), proba_geo)
        qd_geo = test_geo_feat.assign(score=proba_geo).rename(columns={"label": "is_match"})
        rk_geo = evaluate.ranking_metrics([g for _, g in qd_geo.groupby("query_id")])
        out["linkage_cross_state_generalization"] = {**pw_geo, **rk_geo, "test_states": test_states}

    # ---------------- write outputs ----------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "full_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Done. Results written to {OUTPUTS_DIR / 'full_results.json'}")


if __name__ == "__main__":
    main()
