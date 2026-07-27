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

MODEL_DESCRIPTIONS: dict[str, str] = {
    "Expression": "Standalone — log1p(RSEM) gene expression, target-specific selected gene panel.",
    "Mutation": "Standalone — binary mutation matrix + tumor mutational burden (TMB).",
    "Histopathology": "Standalone — 2048-D ResNet50 slide embedding (mean-pooled per patient).",
    "3-Modality": "Fusion — Expression + Mutation + Clinical (late-fusion stacking).",
    "4-Modality": "Fusion — Expression + Mutation + Clinical + Histopathology.",
    "Compare All": "Runs all five backends above and compares outcomes side-by-side.",
}

MODELS_OVERVIEW = (
    "**Five prediction backends:** three standalone models (Expression, Mutation, Histopathology) "
    "and two multimodal stacks (3-Modality, 4-Modality). "
    "**Clinical data is not a standalone model** — it is used only as a level-0 branch inside "
    "the multimodal fusion stacks."
)

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

TARGET_TITLES = {
    "OS_STATUS": "Overall Survival (OS)",
    "PFS_STATUS": "Progression-Free Survival (PFS)",
    "Stage": "AJCC Cancer Stage",
}

TARGET_CLASS_HINTS = {
    "OS_STATUS": {
        0: "Patient survived during the follow-up period",
        1: "Death event observed during follow-up",
    },
    "PFS_STATUS": {
        0: "No disease progression during follow-up",
        1: "Progression or recurrence event observed",
    },
    "Stage": {
        1: "Early-stage, localized disease",
        2: "Locally advanced disease",
        3: "Regional spread (lymph node involvement)",
        4: "Advanced / metastatic disease",
    },
}

POSITIVE_CLASS_LABEL = {
    "OS_STATUS": "Deceased",
    "PFS_STATUS": "Progression",
}


def human_class_label(target: str, class_value: str | int) -> str:
    try:
        key = int(class_value)
    except (ValueError, TypeError):
        return str(class_value)
    return TARGET_LABEL_MAPS.get(target, {}).get(key, f"Class {class_value}")


def class_probability_caption(target: str, class_value: str | int, probability: float) -> str:
    label = human_class_label(target, class_value)
    try:
        hint = TARGET_CLASS_HINTS.get(target, {}).get(int(class_value), "")
    except (ValueError, TypeError):
        hint = ""
    if hint:
        return f"{label} — {hint}: {probability:.1%}"
    return f"{label}: {probability:.1%}"


def build_interpretation(target: str, predicted_label: int, probabilities: dict[str, float]) -> str:
    pred_name = human_class_label(target, predicted_label)
    title = TARGET_TITLES.get(target, target)

    if target == "OS_STATUS":
        p_death = probabilities.get("1", probabilities.get("Deceased", 0.0))
        p_alive = probabilities.get("0", probabilities.get("Alive", 0.0))
        return (
            f"**{title}:** The model's best estimate is **{pred_name}**. "
            f"Estimated chance of **death during follow-up** is **{p_death:.1%}**; "
            f"chance of **survival (alive)** is **{p_alive:.1%}**."
        )
    if target == "PFS_STATUS":
        p_prog = probabilities.get("1", probabilities.get("Progression", 0.0))
        p_no = probabilities.get("0", probabilities.get("No Progression", 0.0))
        return (
            f"**{title}:** The model's best estimate is **{pred_name}**. "
            f"Estimated chance of **disease progression** is **{p_prog:.1%}**; "
            f"chance of **no progression** is **{p_no:.1%}**."
        )
    # Stage (multiclass)
    lines = [
        f"**{title}:** The model's best estimate is **{pred_name}**.",
        "Estimated probabilities by stage:",
    ]
    for cls, prob in sorted(probabilities.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):
        name = human_class_label(target, cls)
        hint = TARGET_CLASS_HINTS.get(target, {}).get(int(cls), "")
        if hint:
            lines.append(f"- **{name}** ({hint}): {prob:.1%}")
        else:
            lines.append(f"- **{name}**: {prob:.1%}")
    return "\n".join(lines)

RISK_THRESHOLDS = {"low": 0.33, "medium": 0.66}

DISCLAIMER = (
    "This tool is for research and educational decision support only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment."
)

APP_VERSION = "0.1.0"
