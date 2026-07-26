"""ONCOLENS Streamlit app configuration."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"
FEATURE_STORE = DATA_DIR / "feature_store"
PROCESSED_DIR = DATA_DIR / "processed_data"
CLINICAL_PATH = PROCESSED_DIR / "mutation_data_processed" / "selected_clinical.csv"

TARGETS = ["OS_STATUS", "PFS_STATUS", "Stage"]

MODELS = [
    "Expression",
    "Mutation",
    "Histopathology",
    "3-Modality",
    "4-Modality",
]

SINGLE_MOD_STORE = {
    "Expression": FEATURE_STORE / "expression",
    "Mutation": FEATURE_STORE / "mutation",
    "Histopathology": FEATURE_STORE / "histopathology",
}

REGISTRY_FILES = {
    "Expression": FEATURE_STORE / "registry.json",
    "Mutation": FEATURE_STORE / "registry_mutation.json",
    "Histopathology": FEATURE_STORE / "registry_histopathology.json",
    "3-Modality": FEATURE_STORE / "registry_multimodal.json",
    "4-Modality": FEATURE_STORE / "registry_multimodal_v4.json",
}

MULTIMODAL_3_DIR = FEATURE_STORE / "multimodal_model"
MULTIMODAL_4_DIR = FEATURE_STORE / "multimodal_model_v4"

CLINICAL_FEATS_PER_TARGET = {
    "OS_STATUS": [
        "HISTORY_NEOADJUVANT_TRTYN",
        "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
        "PATH_M_STAGE",
        "PATH_N_STAGE",
        "PATH_T_STAGE",
        "PRIOR_DX",
        "RADIATION_THERAPY",
    ],
    "PFS_STATUS": [
        "HISTORY_NEOADJUVANT_TRTYN",
        "PATH_M_STAGE",
        "PATH_N_STAGE",
        "PATH_T_STAGE",
        "PRIOR_DX",
        "RADIATION_THERAPY",
    ],
    "Stage": [
        "HISTORY_NEOADJUVANT_TRTYN",
        "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
        "PRIOR_DX",
        "RADIATION_THERAPY",
    ],
}

OS_LABELS = {0: "Alive", 1: "Deceased"}
PFS_LABELS = {0: "No Progression", 1: "Progression"}
STAGE_LABELS = {1: "Stage I", 2: "Stage II", 3: "Stage III", 4: "Stage IV"}

TARGET_LABEL_MAPS = {
    "OS_STATUS": OS_LABELS,
    "PFS_STATUS": PFS_LABELS,
    "Stage": STAGE_LABELS,
}

RISK_THRESHOLDS = {"low": 0.33, "medium": 0.66}

DISCLAIMER = (
    "This tool is for research and educational decision support only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment."
)

APP_VERSION = "0.1.0"
