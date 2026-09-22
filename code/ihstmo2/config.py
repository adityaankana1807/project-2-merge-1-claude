"""Paths and shared constants."""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = CODE_DIR.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_GEO = ROOT_DIR / "data" / "geo"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_SYNTHETIC = ROOT_DIR / "data" / "synthetic"
OUTPUTS_DIR = CODE_DIR / "outputs"

for _d in (DATA_PROCESSED, DATA_SYNTHETIC, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260922

# Study window for the real NCRB district-IPC panel (annual, 2001-2012 inclusive).
NCRB_YEAR_MIN = 2001
NCRB_YEAR_MAX = 2012

# Crime categories (columns) in ncrb_district_ipc_2001_2012.csv treated as the
# core count series for the regional-risk engine.
IPC_COUNT_COLUMNS = [
    "murder", "attempt_to_murder", "culpable_homicide_not_amounting_to_murder",
    "rape", "kidnapping_abduction", "kidnapping_and_abduction_of_women_and_girls",
    "dacoity", "robbery", "burglary", "theft", "auto_theft",
    "dowry_deaths", "assault_on_women_with_intent_to_outrage_her_modesty",
    "insult_to_modesty_of_women", "cruelty_by_husband_or_his_relatives",
]

# Subset explicitly flagged in NCRB "Crime Against Women" reporting; used for the
# underreporting-adjustment layer (Mathews et al. 2024 document large, spatially and
# temporally uneven reporting shocks specifically in this subset).
CAW_COLUMNS = [
    "rape", "kidnapping_and_abduction_of_women_and_girls", "dowry_deaths",
    "assault_on_women_with_intent_to_outrage_her_modesty",
    "insult_to_modesty_of_women", "cruelty_by_husband_or_his_relatives",
]
