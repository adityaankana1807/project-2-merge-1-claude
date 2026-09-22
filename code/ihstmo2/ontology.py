"""India legal-section ontology: IPC 1860 <-> BNS 2023 crosswalk, canonical crime
classes/families, and an empirical cross-crime offender-versatility compatibility
matrix used to gate candidate linkage across different statute sections.

The BNS (Bharatiya Nyaya Sanhita) replaced the IPC nationwide on 1 July 2024, so any
FIR-linkage system spanning that date must reconcile both regimes rather than assume a
single legal vocabulary — a requirement specific to the current Indian context that is
absent from all UK/US/Swedish case-linkage literature reviewed for this project.

Compatibility priors are informed by the versatile-offending literature (Tonkin &
Woodhams, 2017) but recalibrated in `calibrate_compatibility_from_pairs` against
observed offender-series co-occurrence rather than left as fixed hand-authored
constants (see risk of over-trusting uncalibrated priors, flagged in the prior-system
review of at least one independent implementation).
"""
from __future__ import annotations

from typing import Dict, Tuple

CANONICAL_CLASSES = [
    "HOMICIDE", "SEXUAL_OFFENCE", "ROBBERY", "DACOITY",
    "BURGLARY_HOUSEBREAKING", "THEFT", "VEHICLE_THEFT", "EXTORTION",
    "CYBER_FINANCIAL", "KIDNAPPING_ABDUCTION",
]

CRIME_FAMILIES = {
    "VIOLENT_PERSONAL": ["HOMICIDE", "SEXUAL_OFFENCE", "KIDNAPPING_ABDUCTION"],
    "ACQUISITIVE_PROPERTY": ["BURGLARY_HOUSEBREAKING", "THEFT", "VEHICLE_THEFT"],
    "ACQUISITIVE_VIOLENT": ["ROBBERY", "DACOITY", "EXTORTION"],
    "CYBER_FINANCIAL": ["CYBER_FINANCIAL"],
}
CLASS_TO_FAMILY = {c: fam for fam, cs in CRIME_FAMILIES.items() for c in cs}

# section -> (canonical_class, description)
IPC_SECTIONS: Dict[str, Tuple[str, str]] = {
    "302": ("HOMICIDE", "Murder"),
    "304": ("HOMICIDE", "Culpable homicide not amounting to murder"),
    "307": ("HOMICIDE", "Attempt to murder"),
    "376": ("SEXUAL_OFFENCE", "Rape"),
    "376D": ("SEXUAL_OFFENCE", "Gang rape"),
    "354": ("SEXUAL_OFFENCE", "Assault/criminal force to woman, outrage modesty"),
    "509": ("SEXUAL_OFFENCE", "Word, gesture or act insulting modesty of a woman"),
    "363": ("KIDNAPPING_ABDUCTION", "Kidnapping"),
    "366": ("KIDNAPPING_ABDUCTION", "Kidnapping/abduction of woman to compel marriage"),
    "392": ("ROBBERY", "Robbery"),
    "394": ("ROBBERY", "Voluntarily causing hurt in committing robbery"),
    "395": ("DACOITY", "Dacoity"),
    "396": ("DACOITY", "Dacoity with murder"),
    "397": ("ROBBERY", "Robbery/dacoity with attempt to cause death or grievous hurt"),
    "457": ("BURGLARY_HOUSEBREAKING", "Lurking house-trespass/house-breaking by night"),
    "458": ("BURGLARY_HOUSEBREAKING", "House-breaking by night after preparation for hurt"),
    "459": ("BURGLARY_HOUSEBREAKING", "Grievous hurt while committing lurking house-trespass"),
    "380": ("THEFT", "Theft in a dwelling house"),
    "379": ("THEFT", "Theft"),
    "379_AUTO": ("VEHICLE_THEFT", "Motor vehicle theft"),
    "384": ("EXTORTION", "Extortion"),
    "386": ("EXTORTION", "Extortion by fear of death or grievous hurt"),
    "498A": ("HOMICIDE", "Cruelty by husband or relatives (mapped under CAW umbrella)"),
    "304B": ("HOMICIDE", "Dowry death"),
    "420": ("CYBER_FINANCIAL", "Cheating and dishonest inducement to deliver property"),
    "66C": ("CYBER_FINANCIAL", "Identity theft (IT Act, retained alongside IPC/BNS)"),
    "66D": ("CYBER_FINANCIAL", "Cheating by personation using computer resource"),
}

BNS_SECTIONS: Dict[str, Tuple[str, str]] = {
    "103": ("HOMICIDE", "Murder (BNS 103)"),
    "105": ("HOMICIDE", "Culpable homicide (BNS 105)"),
    "109": ("HOMICIDE", "Attempt to murder (BNS 109)"),
    "64": ("SEXUAL_OFFENCE", "Rape (BNS 64)"),
    "70": ("SEXUAL_OFFENCE", "Gang rape (BNS 70)"),
    "74": ("SEXUAL_OFFENCE", "Outraging modesty (BNS 74)"),
    "79": ("SEXUAL_OFFENCE", "Word, gesture, act insulting modesty of woman (BNS 79)"),
    "137": ("KIDNAPPING_ABDUCTION", "Kidnapping (BNS 137)"),
    "309": ("ROBBERY", "Robbery (BNS 309)"),
    "311": ("ROBBERY", "Robbery with grievous hurt (BNS 311)"),
    "310": ("DACOITY", "Dacoity (BNS 310)"),
    "312": ("DACOITY", "Dacoity with murder (BNS 312)"),
    "331": ("BURGLARY_HOUSEBREAKING", "Lurking house-trespass/house-breaking (BNS 331)"),
    "332": ("BURGLARY_HOUSEBREAKING", "House-breaking with preparation for hurt (BNS 332)"),
    "333": ("BURGLARY_HOUSEBREAKING", "Grievous hurt in house-breaking (BNS 333)"),
    "305": ("THEFT", "Theft in dwelling (BNS 305)"),
    "303": ("THEFT", "Theft (BNS 303)"),
    "303_AUTO": ("VEHICLE_THEFT", "Motor vehicle theft (BNS 303(2))"),
    "308": ("EXTORTION", "Extortion (BNS 308)"),
    "85": ("HOMICIDE", "Cruelty by husband/relatives (BNS 85)"),
    "80": ("HOMICIDE", "Dowry death (BNS 80)"),
    "318": ("CYBER_FINANCIAL", "Cheating (BNS 318)"),
}

IPC_TO_BNS: Dict[str, str] = {
    "302": "103", "304": "105", "307": "109", "376": "64", "376D": "70",
    "354": "74", "509": "79", "363": "137", "366": "137", "392": "309",
    "394": "311", "395": "310", "396": "312", "397": "309", "457": "331",
    "458": "332", "459": "333", "380": "305", "379": "303",
    "379_AUTO": "303_AUTO", "384": "308", "386": "308", "498A": "85",
    "304B": "80", "420": "318",
}
BNS_TO_IPC: Dict[str, str] = {v: k for k, v in IPC_TO_BNS.items() if not k.endswith("_AUTO")}
BNS_TO_IPC["303_AUTO"] = "379_AUTO"

# Hand-authored priors (Tonkin & Woodhams 2017 versatility framing), used only as the
# cold-start default before `calibrate_compatibility_from_pairs` has seen labelled
# offender series; see paper Sec. 5.2 for why priors alone must not be reported as
# empirical findings.
_DEFAULT_COMPAT: Dict[Tuple[str, str], float] = {
    ("BURGLARY_HOUSEBREAKING", "THEFT"): 0.72,
    ("THEFT", "VEHICLE_THEFT"): 0.65,
    ("BURGLARY_HOUSEBREAKING", "VEHICLE_THEFT"): 0.48,
    ("ROBBERY", "DACOITY"): 0.68,
    ("ROBBERY", "EXTORTION"): 0.52,
    ("DACOITY", "EXTORTION"): 0.45,
    ("BURGLARY_HOUSEBREAKING", "ROBBERY"): 0.42,
    ("VEHICLE_THEFT", "ROBBERY"): 0.55,
    ("THEFT", "ROBBERY"): 0.38,
    ("HOMICIDE", "ROBBERY"): 0.30,
    ("HOMICIDE", "DACOITY"): 0.35,
    ("SEXUAL_OFFENCE", "HOMICIDE"): 0.18,
    ("BURGLARY_HOUSEBREAKING", "SEXUAL_OFFENCE"): 0.15,
}
DEFAULT_BASELINE = 0.05


def normalize_legal_section(raw_section: str, regime: str = "AUTO") -> Dict[str, str]:
    """Map a raw IPC/BNS section string to canonical class/family + cross-regime code."""
    clean = str(raw_section).strip().upper().replace("SECTION", "").replace("SEC", "").replace(".", "").strip()
    if regime in ("IPC", "AUTO") and clean in IPC_SECTIONS:
        cls, desc = IPC_SECTIONS[clean]
        return {
            "input_section": clean, "detected_regime": "IPC",
            "canonical_class": cls, "crime_family": CLASS_TO_FAMILY[cls],
            "ipc_section": clean, "bns_section": IPC_TO_BNS.get(clean, "N/A"),
            "description": desc,
        }
    if regime in ("BNS", "AUTO") and clean in BNS_SECTIONS:
        cls, desc = BNS_SECTIONS[clean]
        return {
            "input_section": clean, "detected_regime": "BNS",
            "canonical_class": cls, "crime_family": CLASS_TO_FAMILY[cls],
            "ipc_section": BNS_TO_IPC.get(clean, "N/A"), "bns_section": clean,
            "description": desc,
        }
    return {
        "input_section": clean, "detected_regime": "UNKNOWN",
        "canonical_class": "THEFT", "crime_family": "ACQUISITIVE_PROPERTY",
        "ipc_section": "379", "bns_section": "303", "description": "Unclassified offence",
    }


def get_cross_crime_compatibility(class_a: str, class_b: str,
                                   learned: Dict[Tuple[str, str], float] | None = None) -> float:
    """Compatibility score in [0, 1]; prefers `learned` (fit from data) over the prior."""
    if class_a == class_b:
        return 1.0
    table = learned if learned is not None else _DEFAULT_COMPAT
    for pair in ((class_a, class_b), (class_b, class_a)):
        if pair in table:
            return table[pair]
    if learned is not None:
        for pair in ((class_a, class_b), (class_b, class_a)):
            if pair in _DEFAULT_COMPAT:
                return _DEFAULT_COMPAT[pair]
    return DEFAULT_BASELINE


def calibrate_compatibility_from_pairs(pairs) -> Dict[Tuple[str, str], float]:
    """Re-estimate cross-crime compatibility as Laplace-smoothed co-occurrence rate.

    `pairs`: iterable of (class_a, class_b, is_same_offender: bool) observed within the
    *training* portion of an offender-disjoint split only (never test/val), matching the
    Laplace-smoothed compatibility-prior fitting pattern independently validated in one
    of the four prior implementations' evaluation harness.
    """
    counts: Dict[Tuple[str, str], list] = {}
    for a, b, linked in pairs:
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        c = counts.setdefault(key, [0, 0])
        c[0] += int(bool(linked))
        c[1] += 1
    return {key: (pos + 1) / (n + 2) for key, (pos, n) in counts.items()}
