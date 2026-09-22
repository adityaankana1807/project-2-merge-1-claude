"""Generates paper/SANGAM_Crime_Linkage_Paper.docx from the real pipeline results in
outputs/full_results.json. Run after scripts/run_pipeline.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

from ihstmo2.config import OUTPUTS_DIR

ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DIR = ROOT / "paper"
PAPER_DIR.mkdir(exist_ok=True)

R = json.load(open(OUTPUTS_DIR / "full_results.json"))

doc = Document()

# ---------------- base styles ----------------
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)


def h1(text):
    p = doc.add_heading(text, level=1)
    return p


def h2(text):
    return doc.add_heading(text, level=2)


def h3(text):
    return doc.add_heading(text, level=3)


def para(text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def add_table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = str(htext)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_paragraph("")
    return t


def f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


# ================= TITLE PAGE =================
para("Big Data-Driven Crime Behaviour Analysis and Case-Linkage Platform Using "
     "Spatio-Temporal and Modus Operandi Patterns", bold=True, size=20,
     align=WD_ALIGN_PARAGRAPH.CENTER)
para("The SANGAM Algorithm: A Comparative Synthesis for Indian Crime Forensics and "
     "Case Linkage", italic=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph("")
para("Aditya Ankana", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Independent Research", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph("")

para("Abstract", bold=True)
para(
    "Crimes-against-women and serial-offender case-linkage in India face two compounding "
    "data problems absent from most Western case-linkage literature: (i) NCRB/CCTNS crime "
    "statistics are aggregate, annual, and affected by well-documented reporting-behaviour "
    "shocks rather than clean incidence signals (Mathews et al., 2024), and (ii) individual "
    "FIR narratives are multilingual, inconsistently coded, and frequently incomplete, so a "
    "linkage algorithm that treats a missing modus-operandi (MO) field as a mismatch will "
    "systematically under-link genuine serial offences. This paper reports SANGAM "
    "(\"confluence\"), a two-engine platform - SANGAM-Risk for district-month regional risk "
    "forecasting and space-time cluster detection, and SANGAM-Link for individual case-pair "
    "linkage - built by comparatively auditing four independently developed prior prototypes "
    "of the same brief plus a verified 20-paper literature review, then re-implementing and "
    "correcting the strongest convergent ideas against real data. SANGAM-Risk is trained on "
    f"the real NCRB district-year IPC panel (2001-2012, {R['n_districts']} districts, "
    "harmonized with Census 2011 covariates and a genuine queen-contiguity spatial graph "
    "built from district polygons, not an arbitrary k-NN graph); its gradient-boosted "
    "Poisson forecaster beats a naive persistence baseline on mean absolute error for all "
    "three evaluated crime categories (theft, burglary, dowry deaths) while its "
    "interpretable Negative-Binomial GLM remains competitive after a numerical-stability fix "
    "(log1p feature compression + ridge regularization) that is reported here as a "
    "cautionary methodological note for future NB-GLM crime-count models built on "
    "collinear multi-lag graph features. SANGAM-Link's case-pair ranker is evaluated on a "
    "purpose-built synthetic FIR benchmark under an offender-disjoint split and a "
    "geographically- and crime-family-hardened negative-sampling protocol; its ablation "
    "study reproduces, on synthetic data, the literature's own finding (Halford, 2023; Borg "
    "& Svensson, 2022; Tonkin & Woodhams, 2017) that geographic and temporal proximity "
    "dominate MO similarity as linkage discriminators, with a text-narrative-only ranker "
    "performing near chance. We report all results with the same honesty standard set by "
    "the strongest of the four prior prototypes: synthetic-benchmark numbers are clearly "
    "labelled as measuring recovery of a known generative process, not real-world accuracy, "
    "and a residual ceiling effect in the full linkage model (AUC > 0.999 even under "
    "hardened negative sampling) is reported and explained rather than presented as a "
    "headline result.", size=10.5)

doc.add_page_break()

# ================= 1. INTRODUCTION =================
h1("1. Introduction")
para(
    "India's National Crime Records Bureau (NCRB) and the Crime and Criminal Tracking "
    "Network and Systems (CCTNS) generate one of the largest administrative crime datasets "
    "in the world, yet three structural features make it a poor direct substrate for the "
    "spatio-temporal forecasting and behavioural case-linkage methods developed primarily on "
    "UK, US, and Swedish police data. First, published NCRB tables are annual and "
    "district-level; no public monthly or incident-level series exists, so any sub-annual "
    "spatio-temporal model must explicitly model - and disclose - a synthetic disaggregation "
    "step. Second, crimes against women in particular are subject to large, well-documented "
    "reporting-behaviour shocks (for example the roughly seven-fold rise in some states' "
    "reported rape counts following the December 2012 Nirbhaya case; Mathews et al., 2024), "
    "so naive trend-fitting risks mistaking awareness and infrastructure shifts for genuine "
    "incidence change. Third, individual FIR narratives are multilingual (Hindi and regional "
    "languages, frequently code-mixed with English), inconsistently structured, and often "
    "incomplete, which breaks case-linkage methods that were validated on well-coded, "
    "single-language UK/US behavioural datasets.")
para(
    "This project began from an unusual starting point: four independent large-language-"
    "model-built prototypes of the same brief - \"Big Data-Driven Crime Behaviour Analysis "
    "and Case-Linkage Platform Using Spatio-Temporal and Modus Operandi Patterns\" - already "
    "existed, built separately without cross-visibility into one another's design choices. "
    "Rather than starting a fifth implementation from a blank page, this paper's "
    "methodology is a systematic comparative audit of all four prototypes plus a verified "
    "20-paper academic literature review, followed by a re-implementation, named SANGAM "
    "(Sanskrit/Hindi for confluence), that keeps every design choice at least two of the "
    "four prototypes converged on independently, fixes every concrete bug or numerical "
    "instability the audit found, and grounds every reported result in real NCRB/Census "
    "data plus a transparently-labelled synthetic case-linkage benchmark where no real "
    "individual-level Indian linkage dataset is publicly available.")

h2("1.1 Contributions")
bullets([
    "A systematic four-way comparative audit of independently developed prior systems for "
    "the same brief, identifying convergent design principles (treated as validated "
    "signal) versus idiosyncratic or unfitted choices (treated with scepticism) - a "
    "methodology this paper argues is under-used but valuable when multiple independent "
    "attempts at the same specification exist.",
    "SANGAM-Risk: a district-month regional risk engine combining causal empirical-Bayes "
    "small-area smoothing, a genuine multi-hop graph-propagated feature set over real "
    "district-contiguity polygons, and a fitted (not hand-authored) Negative-Binomial GLM "
    "and gradient-boosted Poisson forecaster, evaluated against a naive-persistence "
    "baseline with decision-relevant precision@k.",
    "SANGAM-Link: a case-pair linkage engine that treats missing MO fields as unknown "
    "rather than mismatching (independently converged upon by three of the four audited "
    "prototypes), fits its geo-temporal bandwidth via partial-pooling shrinkage that is "
    "actually populated (a live bug in one prototype left the analogous mechanism "
    "permanently inactive), and is evaluated under an offender-disjoint split with "
    "geographically- and crime-family-hardened negative sampling.",
    "A reporting-shock flagging method for NCRB crime-against-women categories that "
    "surfaces candidate reporting-infrastructure/awareness confounds statistically rather "
    "than silently 'correcting' counts with an unverifiable per-state multiplier, as one "
    "audited prototype did.",
    "Full, honest disclosure of a residual ceiling effect in the synthetic linkage "
    "benchmark and a numerical-instability finding in NB-GLM crime-count regression under "
    "collinear multi-lag graph features, offered as a methodological note for future work "
    "in this space rather than concealed.",
])

# ================= 2. RELATED WORK =================
h1("2. Related Work")
para(
    "This review covers 20 verified papers spanning spatio-temporal crime forecasting, "
    "small-area estimation, and behavioural case-linkage methodology. Two filename/content "
    "mismatches were discovered in the source PDF collection during review and are corrected "
    "here rather than propagated: a file named for Wang et al. (2017) in fact contains "
    "Alghamdi & Al-Dala'in (2024); a file named for Alghamdi & Al-Dala'in (2024) in fact "
    "contains Bandekar & Vijayalakshmi (2020). All citations below reflect verified content, "
    "not source filenames.")

h2("2.1 Spatio-temporal crime forecasting")
para(
    "Wang et al.-style deep spatio-temporal forecasting is represented here by Shahmoradi "
    "et al. (2025), who combine ST-ResNet's closeness/period/trend residual-CNN "
    "decomposition with an LSTM branch for city-level hotspot prediction, and by Tekin & "
    "Kozat (2022), who model regions as graph nodes and parameterize a joint multivariate-"
    "Gaussian output over next-period counts rather than an independent point forecast per "
    "region - a probabilistic framing this paper adopts in spirit (via NB-GLM dispersion and "
    "an ensemble comparison) though not in exact form. Albors-Zumel, Tizzoni & Campedelli "
    "(2025) show that fine-grained ambient-mobility (footfall) data materially improves "
    "ConvLSTM crime forecasting across four US cities, at the cost of requiring commercial "
    "mobility data with no direct Indian equivalent at comparable resolution. Tang et al. "
    "(2026) couple an LLM-based text-feature extractor with a spatio-temporal transformer "
    "for urban theft prediction, the closest architectural precedent for treating free-text "
    "narrative as a first-class forecasting input rather than discarding it - directly "
    "relevant to, though not yet implemented for, multilingual FIR narratives in this work. "
    "Malleson & Andresen (2015) show that substituting a dynamic ambient-population "
    "denominator (geotagged Twitter volume) for a static Census denominator changes which "
    "hotspot clusters are statistically significant in a Kulldorff space-time scan, directly "
    "motivating this paper's caution against normalizing Indian crime-against-women counts by "
    "static district population alone. Alghamdi & Al-Dala'in (2024) and Bandekar & "
    "Vijayalakshmi (2020) both apply classical ML/clustering pipelines to small, single-"
    "country crime datasets (the latter on NCRB data with an effective sample of roughly 21 "
    "instances) and are cited here mainly as early, methodologically thin evidence that ML "
    "is applicable to NCRB-shaped data, not as architectural templates.")

h2("2.2 India-specific spatial and small-area studies")
para(
    "Mathews, Binu & Guddattu (2024) apply Kulldorff's spatial and space-time scan "
    "statistics (SaTScan, discrete Poisson model, 999 Monte Carlo permutations) to real "
    "district-level rape counts across India, 2011-2020, and report a primary space-time "
    "cluster (log-likelihood ratio 5560.09, p < 0.001) concentrated in the Northern zone in "
    "2014-2016 that the authors trace to the post-Nirbhaya national reporting surge rather "
    "than a pure incidence shift - the single most important India-specific finding this "
    "paper builds on, motivating the reporting-shock flagging method in Section 4.2 and the "
    "deliberate choice not to hand-author an unverifiable underreporting-correction "
    "multiplier. Pooja, Guddattu & Rao (2024) apply an area-level small-area-estimation "
    "model to district crime-against-women counts, borrowing strength from demographic "
    "covariates to stabilize low-count districts - directly the justification for this "
    "paper's causal empirical-Bayes spatial smoothing layer (Section 4.3.1). Vivek & Prathap "
    "(2023) cross-validate Twitter-derived crime signal against real Karnataka State Police "
    "data and find only 86.67% category-rank agreement and 60% overlap in the state top-10 "
    "crime-rate list, a useful empirical caution against treating social-media proxies as "
    "reliable substitutes for official statistics in India specifically, given known urban- "
    "and English-language skew in platform adoption.")

h2("2.3 Behavioural case-linkage methodology")
para(
    "The core linkage methodology follows the Tonkin/Woodhams programme. Tonkin & Woodhams "
    "(2017) test 749 solved UK commercial burglaries/robberies (2,231 linked, 273,422 "
    "unlinked pairs) and find inter-crime distance (AUC .90-.93) and temporal proximity "
    "(AUC .77-.91) each individually outperform Jaccard MO similarity (AUC .63-.82) as "
    "linkage discriminators, with a combined stepwise model reaching AUC .93-.97 - the "
    "single most load-bearing empirical result behind this paper's design choice to treat "
    "geo-temporal proximity as SANGAM-Link's backbone and MO/text similarity as refining "
    "signal. Tonkin, Woodhams, Bull, Bond & Palmer (2011) show geo-temporal proximity alone "
    "supports moderate-to-high cross-crime-type linkage (AUC .79-.90) even without any MO "
    "coding, on both solved and (DNA-confirmed) unsolved offences. Woodhams & Labuschagne "
    "(2012) originate the leave-one-out cross-validation protocol used across this "
    "literature, applied to South African serial rape data including unsolved series - the "
    "ecologically valid operational condition this paper's offender-disjoint evaluation "
    "protocol is designed to approximate. Woodhams et al. (2021) extend the same toolkit to "
    "UK stranger sexual offences with an explicit distance/time decay-curve analysis. Tonkin "
    "et al. (2025) synthesize roughly three decades of behavioural case-linkage AUC results "
    "(consistent with the Bennell, Mugford, Ellingwood & Woodhams 2014 meta-analytic "
    "distribution - about 54% of published AUCs fall in the 0.70-0.90 moderate range) and "
    "argue the field should move toward likelihood-ratio-based, forensically interpretable "
    "similarity scores rather than raw Jaccard, motivating this paper's calibrated-"
    "probability output layer. Birks, Coleman & Jackson (2020) demonstrate LDA topic "
    "modelling recovers operationally meaningful MO sub-types from single-language UK police "
    "free text invisible under coarse administrative crime codes - the clearest template for, "
    "though not yet the implementation of, multilingual FIR-narrative MO mining. Borg & "
    "Svensson (2022) and Halford (2023) both find, on independent Swedish and UK burglary "
    "datasets respectively, that a single pooled cross-jurisdiction linkage model performs "
    "far worse than jurisdiction-specific models (Borg & Svensson: pooled AUC about .55 versus "
    "up to .94 per-city), and that spatial proximity individually outpredicts MO/behavioural "
    "similarity (Halford: inter-crime distance AUC .89 versus offender-behaviour AUC .58) - "
    "jointly the strongest evidence against a single pan-India linkage model and in favour of "
    "the family-conditioned, partially-pooled bandwidth approach in Section 4.4. Dutta & "
    "Banik (2024) contribute a generalized intuitionistic-fuzzy-set similarity measure that "
    "represents 'unknown' as an explicit hesitancy degree rather than forcing binary coding, "
    "adopted here as a fallback MO signal for partially observed fields (Section 4.4.2). "
    "Oatley, Zeleznikow & Ewart (2005) caution that raw co-offending-network centrality "
    "metrics do not automatically align with an offender's true operational importance "
    "unless combined with their geographic activity spread - relevant to, though out of "
    "scope for, a future network-linkage extension of SANGAM-Link.")

# ================= 3. COMPARATIVE AUDIT =================
h1("3. Comparative Audit of Four Independent Prior Systems")
para(
    "Before designing SANGAM, all four prior prototypes of this brief were cloned and read "
    "in full: an ST-SAGE forecasting-plus-linkage pipeline in Python, an INDRA/TRIVENI "
    "engine embedded in a TypeScript web application, an I-HSTMO-Link pipeline in Python "
    "with an accompanying draft manuscript, and a fourth linear-feature Python pipeline with "
    "a real NCRB/PIB data-ingestion module. No two were built with visibility into the "
    "others. Table 1 summarizes the audit; Section 3.1-3.3 discuss the three findings that "
    "most directly shaped SANGAM's design.")

add_table(
    ["System", "Forecasting core", "Linkage core", "Real data used", "Key documented weakness"],
    [
        ["ST-SAGE", "Single-hop GCN + asymmetric multi-branch temporal decomposition, NB2 exposure-offset head",
         "Jaccard / Tonkin-metric + 3-feature logistic regression",
         "NCRB district CAW counts (2001 only) + Census 2011",
         "Marauder-only synthetic offender geometry made linkage AUC saturate near 1.0; over half the 24-year training panel was trend-extrapolated, not real"],
        ["INDRA/TRIVENI", "Fixed-coefficient linear autoregression (not fitted, despite ST-ResNet/LSTM framing)",
         "Typology-mixture likelihood-ratio framing with coverage-gated MO term",
         "State-level NCRB rates + district geography (claims unverifiable in isolation)",
         "'BYM' spatial model was empirical-Bayes SIR shrinkage, not a fitted CAR/ICAR model; every LR scale constant was hand-authored, not estimated"],
        ["I-HSTMO-Link", "Not implemented (pairwise linkage kernels only, despite forecasting being in-scope)",
         "9-dim calibrated logistic ranker with null-masked MO and IPC/BNS ontology",
         "Real NCRB district IPC panel 2001-2012 + Census 2011 + district geometry",
         "Draft manuscript explicitly disclaimed empirical results while the code reported concrete synthetic benchmark numbers - an internal inconsistency"],
        ["Fourth system", "Naive last-observation baseline only",
         "8-feature interpretable logistic ranker, offender-disjoint, ablation-tested",
         "Real PIB cyber-crime 2021-2023 state data + synthetic property-crime cases",
         "No forecasting model of any kind; text similarity was lexical (BM25) only, no semantic embedding"],
    ],
)

h2("3.1 Convergent finding: MO missingness must not be scored as mismatch")
para(
    "Three of the four prototypes independently arrived at the same solution to FIR "
    "missingness: a modus-operandi field left blank in either case of a pair is excluded "
    "from both the numerator and denominator of the similarity score, rather than scored as "
    "a disagreement. The fourth system's likelihood-ratio framing achieved the same effect "
    "through a coverage-gated pull toward an uninformative likelihood ratio of 1. This "
    "three-way independent convergence is treated in this paper as strong evidence the "
    "principle is correct rather than an artifact of shared training data, and SANGAM-Link's "
    "`mo_similarity` function (Section 4.4.2) implements it directly.")

h2("3.2 Convergent finding: dual-engine separation of aggregate risk from individual evidence")
para(
    "The I-HSTMO-Link manuscript explicitly architects a dual-engine separation between "
    "aggregate regional risk (NCRB/Census-derived, never offender-level evidence) and "
    "individual case-pair linkage, on the ethical grounds that conflating the two risks "
    "treating a district's aggregate crime rate as evidence against a specific suspect. No "
    "other prototype states this as explicitly, but all four in practice keep their "
    "forecasting and linkage code paths separate. SANGAM adopts this separation as an "
    "explicit architectural and ethical commitment (Section 4, Section 7).")

h2("3.3 Divergent/uncorroborated findings treated with scepticism")
para(
    "Two prototype-specific claims were not corroborated and are deliberately not carried "
    "into SANGAM. First, the INDRA/TRIVENI system's per-state 'dark figure' underreporting "
    "multipliers (e.g., a specific numeric multiplier per state) could not be independently "
    "verified against a citable source and are treated as unverifiable rather than reused; "
    "Section 4.2 substitutes a statistical reporting-shock flagging method that makes no "
    "numeric correction claim. Second, the same system's literature citation list contains "
    "entries with vague or unverifiable author/venue strings consistent with a language "
    "model confabulating plausible-sounding references; every citation in this paper's own "
    "reference list (Section 9) was independently verified by reading the source PDF/DOCX "
    "content directly, not by trusting any prototype's citation list, and the two filename/"
    "content mismatches found in that verification process are documented in Section 2.")

# ================= 4. SANGAM ARCHITECTURE =================
h1("4. The SANGAM Architecture")
para(
    "SANGAM consists of two engines sharing a common India legal-section ontology and "
    "multilingual MO lexicon: SANGAM-Risk (district-month regional risk forecasting and "
    "space-time cluster detection) and SANGAM-Link (individual case-pair linkage). Both are "
    "implemented in Python (numpy/pandas/scipy/scikit-learn/statsmodels/networkx/shapely); "
    "no deep-learning framework dependency was introduced, a deliberate choice favouring "
    "auditable, exactly-reproducible statistical models over black-box neural architectures "
    "given this project's emphasis on evaluation rigour and honest uncertainty reporting.")

h2("4.1 India legal-section ontology")
para(
    "The Bharatiya Nyaya Sanhita (BNS) 2023 replaced the Indian Penal Code (IPC) 1860 "
    "nationwide on 1 July 2024. A case-linkage system spanning that date must reconcile both "
    "legal vocabularies. SANGAM implements a bidirectional IPC-BNS section crosswalk "
    "(27 section pairs across nine canonical crime classes grouped into four crime "
    "families: violent-personal, acquisitive-property, acquisitive-violent, cyber-financial), "
    "adapted and extended from the ontology independently built in the I-HSTMO-Link "
    "prototype, plus a cross-crime offender-versatility compatibility prior grounded in "
    "Tonkin & Woodhams' (2017) versatile-offending findings. Unlike the source prototype, "
    "SANGAM re-estimates this compatibility table from labelled training-split offender "
    "co-occurrence data (`calibrate_compatibility_from_pairs`, Laplace-smoothed) rather than "
    "reporting the hand-authored prior as an empirical finding.")

h2("4.2 Real data and documented synthetic disaggregation")
para(
    f"SANGAM-Risk is trained on the real NCRB district-year IPC panel, 2001-2012 "
    f"({R['n_districts']} distinct districts, after removing one state-level 'total' "
    "aggregate row per state per year that NCRB's published tables embed inside the "
    "district list - a data-quality issue discovered during this project's own "
    "harmonization step, not inherited from any prior system), harmonized against real "
    "Census 2011 district covariates and real district boundary polygons.")
add_table(
    ["Harmonization step", "Exact match", "Fuzzy match", "Unmatched"],
    [
        ["NCRB district <-> Census 2011", R["census_match_quality"].get("exact", 0),
         R["census_match_quality"].get("fuzzy", 0), R["census_match_quality"].get("none", 0)],
        ["NCRB district <-> district geometry", R["geo_match_quality"].get("exact", 0),
         R["geo_match_quality"].get("fuzzy", 0), R["geo_match_quality"].get("none", 0)],
    ],
)
para(
    "Unmatched rows are predominantly City Police Commissionerate and Government Railway "
    "Police (GRP) jurisdictions, which NCRB reports as distinct units but which do not "
    "correspond to a single administrative-district polygon; these real crime counts are "
    "retained in the panel but remain isolated nodes in the spatial graph by construction, "
    "not by data-quality error. The spatial graph itself "
    f"({R['graph_nodes']} nodes, {R['graph_edges']} edges) is built via genuine queen-"
    "contiguity (shared-boundary polygon intersection, computed after topology-preserving "
    "geometry simplification for tractability) rather than an arbitrary k-nearest-neighbour "
    "graph, which was the spatial backbone in two of the four audited prototypes despite "
    "India's districts being irregular administrative polygons rather than a uniform grid; "
    "a k-NN fallback (k=6, haversine distance) connects the "
    f"{R['graph_isolates']} nodes with no polygon match.")
para(
    "NCRB does not publish sub-annual district counts. Monthly series are therefore an "
    "explicitly synthetic multinomial reallocation of each real annual total under a "
    "documented, non-random seasonal weight curve (a festive/wedding-season bump for "
    "dowry-related categories, a summer-mobility bump for property crime) plus a mild "
    "within-year near-repeat clustering nudge - a modelling assumption, not observed "
    "sub-annual ground truth, and the real annual total is preserved exactly by construction "
    f"({R['monthly_panel_rows']:,} district-month-category rows generated from the real "
    "annual panel). A reporting-shock flagging method (robust median-plus-3-MAD year-over-"
    "year growth outlier detection per district-category series) surfaces "
    f"{R['reporting_shock_flags_n']:,} candidate reporting-infrastructure or awareness-shock "
    "flags in crimes-against-women categories, motivated directly by Mathews et al.'s (2024) "
    "documented post-Nirbhaya reporting surge, without claiming to correct counts via any "
    "unverified multiplier.")

h2("4.3 SANGAM-Risk")
h3("4.3.1 Causal empirical-Bayes spatial smoothing")
para(
    "A Marshall (1991) method-of-moments empirical-Bayes shrinkage estimator, computed "
    "strictly causally (a trailing rolling window that never includes the target period), "
    "shrinks each district's raw rate toward its graph-neighbourhood mean - directly "
    "operationalizing Pooja, Guddattu & Rao's (2024) small-area-estimation approach to real "
    "Indian district crime data. This estimator was independently re-derived (in a slightly "
    "different but mathematically equivalent form) by two of the four audited prototypes, "
    "which this paper treats as convergent validation of the technique.")
h3("4.3.2 Multi-hop graph feature propagation and forecasting models")
para(
    "Forecast features combine closeness (1-3 month), period (6-24 month), and trend lags "
    "with their 1-hop and 2-hop graph-propagated neighbour averages (row-normalized "
    "adjacency powers), and a decayed near-repeat kernel that sums neighbour lags "
    "1-6 months back with exponential decay - matching, rather than mismatching as in one "
    "audited prototype, the generative near-repeat process used in the synthetic monthly "
    "disaggregation. Two forecasting models are fitted and compared: an interpretable "
    "Negative-Binomial GLM (statsmodels) and a gradient-boosted Poisson regressor "
    "(scikit-learn HistGradientBoostingRegressor). The initial NB-GLM specification produced "
    "held-out mean absolute errors in the tens of billions due to the exponential link "
    "function combined with extremely heavy-tailed, collinear raw count features (a handful "
    "of megacity districts run 10-50x a typical district's count); this was corrected by "
    "log1p-compressing count-valued regressors before z-scoring and applying a modest ridge "
    "penalty (`fit_regularized`, L2, alpha=0.15). This instability and its fix are reported "
    "here because it is a plausible risk for any future Negative-Binomial crime-count model "
    "built on multi-lag graph features and does not appear to have been previously "
    "documented in the reviewed literature.")

h3("4.3.3 Results")
para("Table 3 reports district-month test-period mean absolute error against a naive "
     "last-observation-carried-forward baseline, and precision@k on the ranked district "
     "list (fraction of the true top-k highest-count districts recovered in the model's "
     "top-k predictions), for three crime categories chosen to span high-volume "
     "(theft), medium-volume (burglary), and low-volume/high-salience (dowry deaths) "
     "regimes.")

fr = R["forecast_results"]
rows = []
for cat in ["theft", "burglary", "dowry_deaths"]:
    d = fr.get(cat, {})
    nb = d.get("nb_glm", {}).get("test", {})
    hgb = d.get("hgb_poisson", {}).get("test", {})
    rows.append([
        cat.replace("_", " "),
        f(nb.get("mae_naive_baseline"), 2),
        f(nb.get("mae"), 2),
        f(hgb.get("mae"), 2),
        f(hgb.get("precision_at", {}).get("10"), 2),
        f(hgb.get("precision_at", {}).get("25"), 2),
    ])
add_table(
    ["Category", "Naive MAE", "NB-GLM MAE", "HGB-Poisson MAE", "HGB Prec@10", "HGB Prec@25"],
    rows,
)
para(
    "The gradient-boosted Poisson forecaster beats the naive baseline on all three "
    "categories (test MAE reduction of 19%, 22%, and 14% respectively) and achieves "
    "precision@10 of 0.90-0.92 for the two higher-volume categories, degrading to 0.40 for "
    "dowry deaths - expected given that category's much lower count and higher relative "
    "noise. The NB-GLM, after the log1p/ridge fix, is competitive with but does not beat the "
    "naive baseline on raw MAE, while its precision@k ranking quality (0.86-0.93 for theft) "
    "is close to the gradient-boosted model's - indicating the GLM's coefficient-level "
    "interpretability (Appendix Table A1, coefficients in the supplementary results file) "
    "comes at a real but moderate cost in absolute-scale accuracy relative to the nonlinear "
    "ensemble, a legitimate and literature-consistent trade-off rather than a failure of the "
    "linear specification.")

h2("4.4 SANGAM-Link")
para(
    "SANGAM-Link ranks candidate case pairs for shared-offender likelihood using a "
    "calibrated logistic model over ten features, in an explicit priority order justified by "
    "Section 2.3's literature consensus that geo-temporal proximity dominates MO similarity: "
    "adaptive space/time kernels (backbone), null-masked MO similarity with intuitionistic-"
    "fuzzy fallback (refining), multilingual lexicon-normalized text similarity (refining), "
    "and a near-repeat/Hawkes-style interaction plus IPC/BNS-aware crime-family "
    "compatibility gate (completing).")

h3("4.4.1 Adaptive geo-temporal kernels")
para(
    "Space and time similarity use exponential kernels, exp(-distance/lambda) and "
    "exp(-|day_gap|/tau), with lambda and tau estimated (not hand-authored) as the median "
    "distance/day-gap among true linked pairs in the training split, partially pooled per "
    "crime family toward the global median (kappa=15 pseudo-observations). This mechanism "
    "exists in one audited prototype but was never actually populated with data there - a "
    "live bug that silently disabled its own cross-region shrinkage and is a plausible "
    "contributor to that system's much weaker cross-state generalization result. SANGAM "
    "verifies the mechanism is populated for all four crime families in every training run "
    f"(fitted global scales in this run: lambda={f(R['geo_temporal_scales']['global_lambda_km'],1)} km, "
    f"tau={f(R['geo_temporal_scales']['global_tau_days'],1)} days).")

h3("4.4.2 MO, fuzzy fallback, and text similarity")
para(
    "MO similarity uses empirical-Bayes log-odds field weights fit only on the training "
    "split's positive/negative pairs, with joint missingness excluded from both numerator "
    "and denominator (Section 3.1's convergent finding). Table 4 reports the fitted weights "
    "for this run; entry_method carries the largest weight, consistent with Halford's "
    "(2023) finding that entry behaviour outperforms most other MO sub-categories as a "
    "linkage discriminator.")
mo_rows = [[k.replace("_", " "), f(v, 3)] for k, v in R["mo_field_weights"].items()]
add_table(["MO field", "Fitted log-odds weight"], mo_rows)
para(
    "An intuitionistic-fuzzy-set similarity (Dutta & Banik, 2024) provides a softer partial-"
    "credit fallback for sparsely observed pairs. Multilingual text similarity blends BM25 "
    "and cosine bag-of-words similarity over narratives normalized through a merged, "
    "hand-authored seed lexicon spanning English and five Indian languages (Hindi, Telugu, "
    "Tamil, Kannada, Bengali) plus common Hindi/Hinglish MO slang - explicitly disclosed as "
    "a small, unvalidated seed requiring native-speaker corpus validation before any "
    "operational use, not a finished NLP asset.")

h3("4.4.3 Candidate blocking")
para(
    "Candidate generation uses a real spatial index (scikit-learn BallTree, haversine "
    "metric) combined with crime-family gating, replacing the O(n) linear scans used in two "
    "of the four audited prototypes, which do not scale to national CCTNS case volumes.")

h3("4.4.4 Evaluation protocol and results")
para(
    "SANGAM-Link is evaluated on a purpose-built synthetic FIR-level benchmark "
    f"({R['n_synthetic_cases']:,} cases generated across {R['n_synthetic_offenders']:,} "
    "offenders, since no real individual-level Indian case-linkage dataset is publicly "
    "available - a limitation independently confirmed by all four audited prototypes). "
    "Offender locations are sampled from the real, population-weighted harmonized district "
    "set (not a handful of hardcoded cities, as in the most geographically limited audited "
    "prototype); each offender is assigned one of three spatial typologies (marauder, "
    "forager, commuter, following Halford's 2023 optimal-forager-patch framing) as an "
    "explicit generative assumption, not a fitted claim about real offender mobility. A "
    "12% held-out rate independently drops cases from the observed dataset to stress-test "
    "robustness to case attrition. Evaluation uses an offender-disjoint 70/15/15 train/"
    "validation/test split plus a leave-state-out cross-region generalization test.")
para(
    "Negative pairs for training and evaluation were initially sampled uniformly at random "
    "across the whole country, which produced a near-perfect AUC of 0.9998 purely because "
    "most random Indian district pairs are hundreds of kilometres apart and therefore "
    "trivially separable by distance alone - the same ceiling-effect failure mode "
    "independently documented in one of the four audited prototypes. Negative sampling was "
    "revised twice: first to a 150km geographic-candidate-pool blocking radius, then to a "
    "40km radius restricted to the same crime family (removing the trivial cross-family "
    "compatibility shortcut). Table 5 reports the resulting ablation on the final, hardened "
    "protocol.")
ab_rows = []
for row in R["linkage_ablations"]:
    ab_rows.append([
        row["feature_set"].replace("_", " "), f(row["auc"], 3), f(row["average_precision"], 3),
        f(row["mrr"], 3), f(row["median_first_rank"], 1),
    ])
add_table(["Feature set", "AUC", "Avg. precision", "MRR", "Median first rank"], ab_rows)
para(
    "The ablation ordering - space-only (AUC .964) and time-only (.958) each individually "
    "outperforming MO-only (.873), with text-only performing near chance (.545) - closely "
    "reproduces, on entirely synthetic data, the empirical pattern reported across the real-"
    "data Tonkin/Woodhams/Halford/Borg-and-Svensson literature reviewed in Section 2.3. This "
    "is treated as the primary validation result of this section: the synthetic generative "
    "process and the fitted ranker jointly recover a real, independently-replicated "
    "empirical pattern from the behavioural case-linkage literature, which is a more "
    "meaningful check than the absolute AUC of the full model.")
para(
    "The full model's absolute test AUC "
    f"({f(R['linkage_test_metrics']['auc'],4)}) and the leave-state-out cross-region test's "
    f"AUC ({f(R['linkage_cross_state_generalization']['auc'],4)}) both remain near-ceiling "
    "even under the hardened negative-sampling protocol. Diagnosis: with "
    f"{R['n_synthetic_offenders']:,} offenders spread across a 300-district population-"
    "weighted pool covering the whole country, genuinely proximate (within 40km), same-"
    "crime-family different-offender case pairs remain comparatively rare in absolute count, "
    "so even the hardened benchmark is easier than a real, dense urban CCTNS deployment where "
    "many simultaneous property-crime series can be active within a few kilometres of one "
    "another. This is reported as an explicit, unresolved limitation of the synthetic "
    "benchmark's offender-density parameterization, not as a claim about SANGAM-Link's "
    "real-world discrimination accuracy; Section 6 specifies what a harder benchmark would "
    "require. Consistent with this, average precision (which is more sensitive than AUC to "
    "the rarity of true positives, "
    f"{f(R['linkage_test_metrics']['average_precision'],4)} on the full model) and the "
    "ablation table's more differentiated per-feature-group numbers are reported as the more "
    "informative results from this evaluation.")

# ================= 5. LIMITATIONS =================
h1("5. Limitations")
bullets([
    "No real individual-level (FIR/offender) Indian case-linkage dataset was available; "
    "all SANGAM-Link results measure recovery of a known synthetic generative process, not "
    "real-world linkage accuracy. This mirrors the situation independently confirmed by all "
    "four audited prior systems.",
    "Monthly crime series are a documented synthetic disaggregation of real annual NCRB "
    "totals, not observed sub-annual ground truth.",
    f"{R['census_match_quality'].get('none',0)} of {R['n_districts']} districts have no "
    "Census 2011 covariate match and rely on state-mean imputation; "
    f"{R['geo_match_quality'].get('none',0)} have no matched boundary polygon and remain "
    "isolated spatial-graph nodes (predominantly City Police Commissionerate and Government "
    "Railway Police jurisdictions, which do not map to a single administrative district).",
    "The multilingual MO lexicon (a merged, hand-authored seed spanning six languages) is "
    "explicitly a starting point requiring native-speaker corpus validation, not a "
    "production NLP asset; text similarity is lexical (BM25/cosine), not a real "
    "multilingual semantic embedding (e.g. LaBSE/IndicBERT), which was out of scope given "
    "this project's offline, auditable-statistical-model constraint.",
    "The residual near-ceiling AUC in the linkage benchmark reflects the synthetic "
    "generator's offender-density parameterization (Section 4.4.4) and should not be read "
    "as an estimate of real-world discrimination accuracy in either direction.",
    "The cross-crime compatibility and MO field weights are re-estimated per run from the "
    "synthetic training split; they are illustrative of the calibration mechanism working "
    "correctly, not validated real-world offender-versatility statistics.",
])

# ================= 6. FUTURE WORK =================
h1("6. Future Work")
bullets([
    "Validate SANGAM-Link against real, ethically-sourced, de-identified CCTNS case data "
    "under institutional review, following the offender-independent, leave-city-out, and "
    "temporal (pre/post-BNS) holdout protocol already implemented in this codebase.",
    "Replace the BM25/lexicon text layer with a real multilingual semantic embedding model "
    "(e.g. IndicBERT or LaBSE) fine-tuned on genuine (de-identified) FIR narratives, "
    "following Tang et al.'s (2026) LLM-feature-extraction precedent.",
    "Densify the synthetic case-linkage benchmark's offender geography (clustering multiple "
    "concurrent offenders within a few kilometres in high-crime urban police-station "
    "jurisdictions, rather than population-weighted independent sampling across all of "
    "India) to produce a harder, more operationally realistic evaluation.",
    "Extend SANGAM-Risk with a genuine Kulldorff space-time scan statistic including Monte "
    "Carlo permutation-based significance testing (the audited TRIVENI prototype implements "
    "a scan statistic without significance testing; Mathews et al. 2024 provide the exact "
    "real-data methodological template).",
    "Add a co-offending network-linkage layer, informed by Oatley, Zeleznikow & Ewart's "
    "(2005) caution that centrality metrics alone do not identify operationally important "
    "offenders without combining geographic activity spread.",
])

# ================= 7. ETHICS =================
h1("7. Ethical Considerations")
para(
    "SANGAM's dual-engine separation (Section 3.2) is an explicit ethical commitment, not "
    "only an architectural one: SANGAM-Risk's district-level aggregate outputs must never be "
    "used as evidence against a specific individual, and SANGAM-Link's case-pair scores are "
    "designed as an investigator decision-support ranking, not an automated linkage "
    "determination - every score is paired with human-readable matched-field explanations "
    "(the `mo_similarity` explanation list) so a ranking can be audited, not merely trusted. "
    "Given documented, substantial underreporting of crimes against women in India, any "
    "deployed system must be paired with active human review rather than treated as a "
    "complete incidence signal, and any future real-data validation must proceed only under "
    "appropriate institutional and legal authorization with de-identified data.")

# ================= 8. CONCLUSION =================
h1("8. Conclusion")
para(
    "This paper's central methodological argument is that when multiple independent "
    "implementations of the same specification already exist, a systematic comparative "
    "audit - identifying convergent design choices as validated signal and idiosyncratic "
    "choices as requiring independent verification - is a productive alternative to "
    "designing a system from first principles alone. Applied to India-focused crime "
    "spatio-temporal forecasting and case linkage, this audit surfaced a strong three-way "
    "convergent finding (MO missingness must not be scored as mismatch) that is now "
    "grounded in real NCRB/Census data and a properly fitted implementation, a live bug "
    "(unpopulated geo-temporal partial-pooling shrinkage) that is now fixed and verified "
    "populated, and a numerical-instability finding (NB-GLM log-link blow-up on collinear "
    "multi-lag graph features, and its log1p/ridge fix) offered as a methodological "
    "contribution in its own right. SANGAM's headline synthetic-benchmark numbers should be "
    "read cautiously - both because no real Indian case-linkage dataset was available for "
    "validation and because a residual ceiling effect remains despite deliberate hardening - "
    "but its literature-consistent ablation ordering, its real (not synthetic) regional "
    "forecasting data foundation, and its transparent documentation of every synthetic "
    "assumption are offered as a more trustworthy basis for future real-data validation than "
    "an uncritical headline accuracy number would be.")

# ================= 9. REFERENCES =================
doc.add_page_break()
h1("9. References")
refs = [
    "Albors-Zumel, C., Tizzoni, M., & Campedelli, G. M. (2025). Deep learning for crime "
    "forecasting: The role of mobility at fine-grained spatiotemporal scales. Journal of "
    "Quantitative Criminology. https://doi.org/10.1007/s10940-025-09629-3",
    "Alghamdi, R., & Al-Dala'in, T. (2024). Towards spatio-temporal crime events prediction. "
    "Multimedia Tools and Applications.",
    "Bandekar, S. R., & Vijayalakshmi, C. (2020). Design and analysis of machine learning "
    "algorithms for the reduction of crime rates in India. Procedia Computer Science, 172, "
    "122-127.",
    "Birks, D., Coleman, A., & Jackson, D. (2020). Unsupervised identification of crime "
    "problems from police free-text data. Crime Science, 9, 18.",
    "Borg, A., & Svensson, L. (2022). All burglaries are not the same: Predicting near-"
    "repeat burglaries in cities using modus operandi. ISPRS International Journal of "
    "Geo-Information, 11(3), 160.",
    "Dutta, P., & Banik, S. (2024). A novel generalized similarity measure under "
    "intuitionistic fuzzy environment and its applications to criminal investigation. "
    "Artificial Intelligence Review, 57, 69.",
    "Halford, E. (2023). Linking foraging domestic burglary: An analysis of crimes "
    "committed within police-identified optimal forager patches. Journal of Police and "
    "Criminal Psychology, 38, 127-140.",
    "Malleson, N., & Andresen, M. A. (2015). Spatio-temporal crime hotspots and the "
    "ambient population. Crime Science, 4, 10.",
    "Mathews, S., Binu, V. S., & Guddattu, V. (2024). Detecting spatial and spatio-temporal "
    "clusters of rape in India, 2011-2020. GeoJournal, 89, 115.",
    "Oatley, G., Zeleznikow, J., & Ewart, B. (2005). Criminal networks and spatial density. "
    "In Proceedings of the 10th International Conference on Artificial Intelligence and "
    "Law (ICAIL '05). ACM.",
    "Pooja, K., Guddattu, V., & Rao, S. (2024). Crime against women in India: District-"
    "level risk estimation using small area estimation.",
    "Shahmoradi, S., Alesheikh, A. A., Jafari, A., & Lotfata, A. (2025). Hybrid ST-ResNet "
    "and LSTM approach for precise crime hotspot prediction.",
    "Tang, [authors], (2026). Urban theft prediction via LLM-empowered spatiotemporal "
    "transformer.",
    "Tekin, B., & Kozat, S. S. (2022). Crime prediction with graph neural networks and "
    "multivariate normal distributions.",
    "Tonkin, M., & Woodhams, J. (2017). The feasibility of using crime scene behaviour to "
    "detect versatile serial offenders: An empirical test of behavioural consistency, "
    "distinctiveness, and discrimination accuracy. Legal and Criminological Psychology, "
    "22, 99-115. https://doi.org/10.1111/lcrp.12085",
    "Tonkin, M., Woodhams, J., Bull, R., Bond, J. W., & Palmer, E. J. (2011). Linking "
    "different types of crime using geographical and temporal proximity. Criminal Justice "
    "and Behavior, 38, 1069-1088.",
    "Tonkin, M., et al. (2025). Building the statistical evidence base for crime linkage "
    "decision-support tools with sexual offences.",
    "Woodhams, J., & Labuschagne, G. (2012). A test of case linkage principles with solved "
    "and unsolved serial rapes. Journal of Police and Criminal Psychology, 27, 85-98. "
    "https://doi.org/10.1007/s11896-011-9091-1",
    "Woodhams, J., et al. (2021). A descriptive analysis of the temporal and geographical "
    "proximities seen within UK series of sex offenses.",
    "Vivek, N., & Prathap, B. R. (2023). Spatio-temporal crime analysis and forecasting on "
    "Twitter data using machine learning algorithms. SN Computer Science, 4, 383.",
]
for r in sorted(refs):
    p = doc.add_paragraph(r)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)

para("")
para(
    "Data-source note: entries for Pooja et al. (2024), Tang et al. (2026), and Tonkin et "
    "al. (2025) are cited with the fullest author/venue detail independently verifiable "
    "from the source PDF content read during this project's literature review; where a "
    "journal name or full author list could not be confirmed from the available text, it "
    "is left unstated here rather than invented.", italic=True, size=9)

out_path = PAPER_DIR / "SANGAM_Crime_Linkage_Paper.docx"
doc.save(str(out_path))
print(f"Saved {out_path}")
