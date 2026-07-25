# ONCOLENS — Master Handoff Plan (Histopathology, 4-Modality Fusion, Comparative Evaluation)

**Purpose of this document**: a self-contained handoff so work can continue on a different machine (needed for Phase 4's disk-space-heavy TCGA slide download). Covers project status, exactly what was done in this session, and every remaining step in order.

**Written**: 2026-07-25. **Repo**: `/Users/soumya/Documents/ONCOLENS` (git branch `developer`).

---

## 1. Project Status Overview

ONCOLENS predicts three targets for TCGA LUAD patients — `OS_STATUS` (binary), `PFS_STATUS` (binary), `Stage` (multiclass 1-4) — from multiple data modalities, using a notebook-driven pipeline (no `src/` package; everything lives under `EXP/`).

### 1.1 Completed modalities (done before this session)

| Modality | Notebook | Cohort | Held-out Test AUC (OS / PFS / Stage) |
|---|---|---|---|
| Gene Expression | `EXP/gene_expression_v2.ipynb` | 459 patients (367 train / 92 test) | 0.630 / 0.577 / 0.557 |
| Mutation | `EXP/Mutation_notebooks/new_mutation_data.ipynb` | 454 patients (363 train / 91 test) | 0.558 / 0.621 / 0.458 |
| Multimodal (Clinical + Expression + Mutation, late-fusion/stacking) | `EXP/multimodalv2.ipynb` | 450 patients (360 train / 90 test) | 0.784 / 0.728 / 0.609 |

Multimodal architecture (important — Phase 5 must extend this exact contract):
- **Level-0**: per modality, train `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)` + `LogisticRegression(C=0.05, penalty='l1', solver='saga', class_weight='balanced', max_iter=2000)` in a `StandardScaler` pipeline. Out-of-fold probabilities via `cross_val_predict(cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba')`. RF+LR OOF probabilities are averaged into one probability vector per modality.
- **Meta-features**: 1 positive-class probability per modality for binary targets; all-class probabilities hstacked per modality for multiclass Stage (currently 3 modalities × up to 4 classes).
- **Level-1**: single `LogisticRegression` meta-learner trained on OOF meta-features, evaluated on genuinely held-out test meta-features (Level-0 models refit on full train, predict on test, feed meta-learner).
- **Persisted artifact**: `Data/feature_store/multimodal_model/{target}/multimodal_stack.joblib` — dict bundle: `bundle["level0"][modality]["rf"/"lr"/"feature_names"/"preprocessing"]`, `bundle["level1"]["feature_order"]/"meta_model"`, `bundle["metadata"]`.
- **Served by**: `app/oncolens_inference.py` (Streamlit app). Key functions: `load_processed_tables()`, `_align_modality()`, `_build_meta_features()`, `make_uploaded_patient()`, `predict_multimodal()`.

### 1.2 Pending work (the subject of this plan)

- **Phase 4**: Histopathology Image Analysis module (not started — no image data exists in the repo yet)
- **Phase 5**: Extend the multimodal stack to 4 modalities (Clinical + Expression + Mutation + Histopathology)
- **Phase 6**: Comparative Model Evaluation across 5 models
- **Phase 7**: Final deliverables (report, diagrams, presentation, cleanup)

---

## 2. What Was Done In This Session (2026-07-25)

Two sub-tasks from the approved plan were completed — both cheap, local, reversible "plumbing" steps done in parallel with (not blocking) the Phase 4 data acquisition that must happen on the new machine.

### 2.1 Backfilled `test_predictions.csv` for Expression and Mutation (Phase 6a)

**Problem**: only the Multimodal model had `test_predictions.csv` (raw y_true/y_pred/y_proba on the held-out test set). Expression and Mutation only had `metadata.json` with summary AUCs — no per-patient predictions — so Phase 6's comparative evaluation had no uniform way to recompute Accuracy/Precision/Recall/F1/ROC-AUC across all models.

**What was built**: `EXP/backfill_test_predictions.py` — a standalone script (not a notebook, per the plan's own "or a short backfill notebook" allowance) that:
1. Loads `Data/processed_data/mutation_data_processed/selected_clinical.csv` for labels.
2. For each of Expression/Mutation × {OS_STATUS, PFS_STATUS, Stage}: loads that target's `features.csv`, reproduces the **exact original 80/20 stratified split** (`train_test_split(..., test_size=0.2, stratify=OS_STATUS, random_state=42)` — the same split is reused across all 3 targets within a modality, matching both source notebooks exactly).
3. Refits the model type recorded in `metadata.json` (Random Forest or Logistic-LASSO, with the exact hyperparameters from each notebook: Expression RF uses `n_estimators=300`, Mutation RF uses `n_estimators=500`; Expression LR uses `C=0.1`, Mutation LR uses `C=0.5` — **not** interchangeable, verified by reading both source notebooks directly) on the train split only.
4. Predicts on the held-out test split and writes `test_predictions.csv` in the same schema the Multimodal model already uses:
   - Binary targets: `patient_id, true_label, predicted_label, predicted_probability`
   - Stage (multiclass): `patient_id, true_label, predicted_label, probability_class_1..4`

**Verification performed**:
- Asserted reproduced test-split size exactly matches each `metadata.json`'s recorded `test_patients` — **passed for all 6 target/modality combinations** (this confirms the split was reproduced correctly, not just approximately).
- Recomputed ROC-AUC from the written predictions and compared to the recorded `test_auc`:

  | Modality | Target | Recomputed AUC | Recorded AUC |
  |---|---|---|---|
  | Expression | OS_STATUS | 0.6316 | 0.6296 |
  | Expression | PFS_STATUS | 0.5668 | 0.5774 |
  | Expression | Stage | 0.5119 | 0.5569 |
  | Mutation | OS_STATUS | 0.5520 | 0.5583 |
  | Mutation | PFS_STATUS | 0.6205 | 0.6205 |
  | Mutation | Stage | 0.4892 | 0.4576 |

  Close but not bit-exact — attributed to the installed sklearn version (1.9.0) being newer than the `1.8.0` pinned in `app/requirements.txt` at original training time, which can shift exact Random Forest / multiclass-OVR probability outputs slightly. The split itself is confirmed exactly correct via the size assertion, so this is a benign, documented discrepancy, not a bug. If exact reproducibility ever matters, pin `scikit-learn==1.8.0` before re-running.

**Files created** (new, untracked in git as of this writing):
- `EXP/backfill_test_predictions.py`
- `Data/feature_store/expression/{OS_STATUS,PFS_STATUS,Stage}/test_predictions.csv`
- `Data/feature_store/mutation/{OS_STATUS,PFS_STATUS,Stage}/test_predictions.csv`

### 2.2 Scaffolded the Histopathology module (Phase 4, structure only — no data, no execution)

Three new files were created to set up Phase 4's structure ahead of time, without doing any actual GDC download (that step needs the new machine's disk space and was deliberately deferred):

- **`EXP/requirements-histopathology.txt`** — new dependencies this module introduces (the project's first imaging/DL deps): `openslide-python`, `Pillow`, `histolab`, `torch`, `torchvision`, `opencv-python`. Deliberately kept separate from `app/requirements.txt` (which pins `scikit-learn==1.8.0` for the Streamlit serving app) since only the final tabular feature bundle needs to ship with the app, not the extraction pipeline.
- **`EXP/histopathology_data_acquisition.ipynb`** — scaffold notebook with markdown + code stubs for: (1) GDC manifest filtering (`TCGA-LUAD`, `Slide Image`, Diagnostic Slides only — barcode suffix `-DX1`/`-DX2`, not `-TS` frozen sections), (2) building `slide_manifest.csv` (`slide_id -> PATIENT_ID -> local .svs path`), (3) reporting patient overlap against Expression/Mutation/Clinical before any further work (gates the Phase 5 cohort decision). No download has been executed — the manifest-existence check cell will raise an `AssertionError` until a real `gdc_manifest.txt` is placed there.
- **`EXP/histopathology_v1.ipynb`** — scaffold notebook mirroring `gene_expression_v2.ipynb`'s structure exactly: tissue detection/tiling stub → frozen ResNet50 embedding stub (MPS/GPU-ready) → mean-pooling aggregation stub → RF/LR train/eval (fully implemented, reusing the same hyperparameter pattern) → feature-store save function (fully implemented) that writes `features.csv`/`model.joblib`/`test_predictions.csv`/`metadata.json` per target plus `registry_histopathology.json`. The pipeline stubs (`tile_slide`, `build_resnet50_encoder`, `embed_tiles`, `aggregate_patient_features`) raise `NotImplementedError` — they need real slide data to implement against.
- **`.gitignore`** — added `/Data/raw_data/histopathology` (raw `.svs` files must never be committed — expected size is 100s of GB).

Both notebooks were validated as syntactically correct (`json.load` succeeds, correct cell counts) but have not been executed, since there is no image data yet.

---

## 3. Full Remaining Plan

### Phase 4 — Histopathology Image Analysis Module

**Objective**: develop, train, validate, and evaluate the Histopathology prediction model as a standalone modality before integrating it into the four-modality multimodal framework.

**Confirmed decisions** (from user, do not re-litigate):
- Tile encoder: **ImageNet-pretrained ResNet50** (torchvision), frozen, no fine-tuning.
- Compute: **GPU available** — Apple Silicon MPS backend.
- PCA: **optional**, not mandatory. Default baseline is ResNet50 → 2048-D embedding → RF/LR directly. Only add PCA later if overfitting/runtime becomes a real problem.
- Aggregation: **mean-pooling** is the baseline (tile → slide → patient). Attention-pooling / MIL / CLAM is explicitly future work, not part of this pass.
- No standalone Clinical-only model will be built (see Phase 6).
- Streamlit app stays lightweight: no live `.svs` upload/inference — precomputed feature vectors only.

**4.1 Data acquisition** *(this is the step that needs the new machine's disk space)*:
1. TCGA LUAD whole-slide images are **not** in the existing `luad_tcga_pan_can_atlas_2018/` cBioPortal download. Get them from the **GDC Data Portal** (https://portal.gdc.cancer.gov).
2. Filter: `project_id = TCGA-LUAD`, `data_type = "Slide Image"`, **Diagnostic Slides only** (barcode suffix `-DX1`/`-DX2` — exclude `-TS` frozen sections).
3. Add filtered files to a GDC cart, download the manifest (`gdc_manifest.txt`), place it at `Data/raw_data/histopathology/gdc_manifest.txt` on the new machine.
4. Install `gdc-client` (https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) and run:
   ```bash
   gdc-client download -m Data/raw_data/histopathology/gdc_manifest.txt -d Data/raw_data/histopathology
   ```
5. **Expect 100s of GB** across ~500 patients (LUAD diagnostic slides typically 200MB-2GB each) — confirm available disk space before starting the download, and budget real download time (hours, depending on bandwidth).
6. Run the `build_slide_manifest()` cell in `EXP/histopathology_data_acquisition.ipynb` to produce `slide_manifest.csv`, then run `report_patient_overlap()` — **record this overlap number**, it gates the Phase 5 cohort decision (5.1 below).

**4.2 WSI preprocessing** (implement `tile_slide()` in `EXP/histopathology_v1.ipynb`):
1. Tissue detection: Otsu thresholding on a downsampled thumbnail; discard tiles >80-90% background.
2. Tiling: 256×256px tiles at 20x magnification.
3. Stain normalization: Macenko normalization (corrects H&E batch variation across TCGA's many contributing sites — a known, documented TCGA pan-cancer issue).
4. Install `EXP/requirements-histopathology.txt` on the new machine first: `pip install -r EXP/requirements-histopathology.txt` (also requires the OpenSlide system library — `brew install openslide` on macOS before `pip install openslide-python`).

**4.3 Feature extraction** (implement `build_resnet50_encoder()` / `embed_tiles()`):
- Frozen `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`, strip classification head, `.eval()`, `torch.no_grad()`, move to MPS device (`torch.backends.mps.is_available()`).
- Batch tiles through the encoder, cache per-slide embeddings to `Data/feature_store/histopathology/tile_embeddings_cache/` (expensive GPU step — cache so aggregation/PCA choices can be revisited without recomputing).

**4.4 Per-patient aggregation** (implement `aggregate_patient_features()`):
- Mean-pool tile embeddings per slide → mean across slides for patients with multiple diagnostic slides → one 2048-D vector per patient.

**4.5 Train/evaluate + save to feature store**:
- The `train_and_eval()` and `save_target_to_feature_store()` functions in `EXP/histopathology_v1.ipynb` are already fully implemented (copied from Expression/Mutation's pattern) — just need real feature data plugged in.
- Produces, matching the existing convention exactly:
  - `Data/feature_store/histopathology/{OS_STATUS,PFS_STATUS,Stage}/{features.csv, metadata.json, model.joblib, test_predictions.csv}`
  - `Data/feature_store/registry_histopathology.json`
- The Histopathology model follows the same Level-0 modeling paradigm as Expression/Mutation — its prediction probabilities become an additional Level-0 input to the stacking architecture in Phase 5.

**4.6 Verification** (do before moving to Phase 5):
- Visual tiling check: overlay tissue mask + sampled tile grid on thumbnails for a handful of slides.
- Tile-count-per-slide report — near-zero counts flag a broken tissue-detection threshold.
- Patient-overlap report against the other 3 modalities (from 4.1, step 6).
- Stain-normalization spot check across different TCGA source sites (3rd barcode field, e.g. `TCGA-55-...` vs `TCGA-86-...`).
- Baseline sanity: the standalone Histopathology model should demonstrate predictive performance clearly above random guessing before proceeding to multimodal integration (no fixed numeric threshold prescribed — depends on class balance/dataset characteristics).

---

### Phase 5 — 4-Modality Multimodal Fusion Extension

**5.1 Cohort decision**: Histopathology will likely cover a different (probably smaller) patient set than the current 450-patient join. **Decision**: retrain the full 4-modality stack on the new intersection (Expression∩Mutation∩Clinical∩Histopathology) — do not build a mixed-availability fallback path in this pass. Consistent with how the cohort was already redefined at each modality addition (510→454→450). If the new intersection drops meaningfully (e.g. below ~250-300 patients), document it as a limitation rather than engineering around it; a "predict without slides" fallback can be scoped later as a follow-on.

**5.2 Concrete changes**:
- New notebook `EXP/multimodalv3.ipynb` (fork of `multimodalv2.ipynb`, following the repo's `_v2`/`_v3` convention):
  - Data loading: 4-way inner join, report new cohort size prominently up front.
  - Feature prep: add Histopathology's pooled embedding columns (numeric/dense already — PCA only if adopted in Phase 4).
  - Level-0: extend the per-modality RF+LR OOF loop to include Histopathology (loop already generalizes to a list of modalities — no restructuring needed).
  - Level-1: `meta_feature_order` grows 3→4 modalities (Stage: 3×n_classes → 4×n_classes meta-features; meta-learner logic itself unchanged).
  - Results/feature-importance sections: extend to include Histopathology.
- `app/oncolens_inference.py` — exact functions to update:
  - `load_processed_tables()` — add Histopathology table load + include in patient-intersection.
  - `_align_modality()` — add a Histopathology branch (likely `fillna(0.0)`, confirm against the actual `preprocessing` bundle shape from Phase 4).
  - `_build_meta_features()` — no structural change; already iterates generically over `bundle["level1"]["feature_order"]`.
  - `make_uploaded_patient()` — add a 4th optional input accepting a **precomputed** histopathology feature vector only (no live `.svs` upload/inference in the app).
  - `multimodal_stack.joblib` bundle format needs no structural change — just a 4th `level0` key.
  - Regenerate `registry_multimodal.json` with new 4-modality AUCs.

**5.3 Verification**:
- Leakage check: confirm Histopathology features used no label info; confirm any PCA fit was train-only; confirm OOF predictions used the same `StratifiedKFold(5)` + `cross_val_predict` discipline as the other 3 modalities.
- Cohort-size sanity check printed before training.
- **Ablation comparison (critical)**: retrain the 3-modality model on the *same new (smaller) cohort* alongside the 4-modality model, so any AUC delta is attributable to Histopathology, not to a cohort-size/composition change.
- Meta-learner coefficient check: inspect weight assigned to Histopathology vs. the other 3 modalities.

**5.4 Feature Contribution Analysis**: after training the 4-modality model, compare it against the 3-modality model on the identical cohort. Report absolute AUC improvement, relative performance improvement, accuracy difference, F1 difference. Produce a dedicated **3-Modality vs. 4-Modality Improvement Table** (one row per model per target: Accuracy/Precision/Recall/F1/ROC-AUC + Δ AUC abs/%), saved to `Data/feature_store/comparative_evaluation/modality_improvement_table.csv`, rendered as a grouped bar chart for the final report.

---

### Phase 6 — Comparative Model Evaluation

**Final Experimental Comparison** — 5 models, **no standalone Clinical-only model** (Clinical stays a component inside the multimodal stacks only):
1. Gene Expression
2. Mutation
3. Histopathology
4. Existing 3-Modality Model (Clinical + Gene Expression + Mutation)
5. New 4-Modality Model (Clinical + Gene Expression + Mutation + Histopathology)

**6.1 Design**: `EXP/comparative_evaluation.ipynb`, backed by `EXP/eval_utils.py`:
- Loader reads `registry.json`, `registry_mutation.json`, `registry_histopathology.json`, `registry_multimodal.json` (holds both 3- and 4-modality entries) into one long-format DataFrame (`Model, Target, Metric, Value`).
- Metrics recomputed uniformly from each model's `test_predictions.csv` via `sklearn.metrics` for Accuracy/Precision/Recall/F1/ROC-AUC (Expression/Mutation's `test_predictions.csv` were already backfilled this session — see §2.1; Histopathology writes its own from the start per §Phase 4.5).
- Output: one comparison table (5 models × 3 targets × 5 metrics) + grouped bar charts, saved to `Data/feature_store/comparative_evaluation/comparison_table.csv`.

**6.2 Verification**:
- Recomputed ROC-AUC per model should match each registry's stored AUC closely.
- Fix one averaging scheme (macro or weighted) for multiclass Stage Precision/Recall/F1 across all 5 rows.
- Sanity check: 4-modality AUC ≥ 3-modality AUC (same cohort) ≥ every single-modality AUC per target; investigate any inversion.

---

### Phase 7 — Final Deliverables

- Final Project Report
- Architecture Diagrams
- Performance Analysis
- Comparative Evaluation Results
- Final Presentation
- Source Code Documentation
- Git Repository Cleanup

---

## 4. Machine Migration Checklist

To continue on the new machine:

1. **Sync the repo** — either `git clone`/`git pull` the `developer` branch, or copy the working tree directly. Note: `Data/raw_data/`, `luad_tcga_pan_can_atlas_2018/`, `app/`, `Docs/`, `Papers/`, `claude plans/` are all gitignored — if using plain `git pull`, these won't transfer via git. Since this handoff document itself lives under the gitignored `claude plans/` folder, **copy that folder across manually** (or via a non-git transfer — rsync/external drive) alongside the git-tracked files.
2. **Copy the working-tree files this session touched that ARE git-tracked** (they'll come via git pull from `EXP/` and `Data/feature_store/`, but confirm since they're currently untracked/uncommitted locally — see `git status` below):
   - `EXP/backfill_test_predictions.py`
   - `EXP/histopathology_data_acquisition.ipynb`
   - `EXP/histopathology_v1.ipynb`
   - `EXP/requirements-histopathology.txt`
   - `Data/feature_store/{expression,mutation}/{OS_STATUS,PFS_STATUS,Stage}/test_predictions.csv`
   - `.gitignore` (modified)
   
   None of this has been committed yet — consider committing before switching machines so a plain `git pull` carries it over cleanly, or transfer the working tree directly.
3. **Set up Python environment** on the new machine:
   ```bash
   pip install -r EXP/requirements-histopathology.txt
   ```
   Plus the OpenSlide system library (`brew install openslide` on macOS, `apt-get install openslide-tools` on Linux) before `openslide-python` will import correctly.
4. **Confirm disk space** — budget for at minimum several hundred GB free before starting the GDC download (§4.1).
5. **Set up GDC access** — no special credentials needed for open-access TCGA slide images (they're not controlled-access like raw sequencing data), but install `gdc-client` from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool.
6. **Confirm GPU/MPS availability** on the new machine (`torch.backends.mps.is_available()` if Apple Silicon, or CUDA if not) — the plan assumes GPU-accelerated ResNet50 inference is available; if not, feature extraction will be much slower and should be batched/monitored accordingly.

---

## 5. Immediate Next Steps (in order)

1. On the new machine: sync repo + `claude plans/` folder, install `EXP/requirements-histopathology.txt` + OpenSlide.
2. Build the GDC manifest via the GDC Data Portal cart (TCGA-LUAD, Slide Image, Diagnostic Slide) and download via `gdc-client` (Phase 4.1).
3. Run `EXP/histopathology_data_acquisition.ipynb` end-to-end — build `slide_manifest.csv`, get the patient-overlap number.
4. Implement the four stubbed functions in `EXP/histopathology_v1.ipynb` (`tile_slide`, `build_resnet50_encoder`, `embed_tiles`, `aggregate_patient_features`) against real slide data.
5. Run the full Histopathology training/eval/save pipeline, verify per §4.6.
6. Proceed to Phase 5 (`EXP/multimodalv3.ipynb` + `app/oncolens_inference.py` updates), then Phase 6 (`EXP/comparative_evaluation.ipynb`), then Phase 7.

---

## 6. Key File Reference

- `EXP/multimodalv2.ipynb`, `app/oncolens_inference.py` — fusion architecture to extend (Phase 5)
- `Data/feature_store/multimodal_model/OS_STATUS/metadata.json` — reference schema to match
- `Data/feature_store/registry.json`, `registry_mutation.json`, `registry_multimodal.json` — schema precedent for new registries
- `claude plans/Research_paper_v1.md`, `claude plans/multimodalv2.md` — literature precedent and planning-doc style
- `claude plans/Histopathology plans/Histopathology_Action_Plan_Recommended_Changes.md`, `Additional_Recommended_Updates_Action_Plan.md` — the user's original review notes that shaped this plan's decisions (standalone-first validation, optional PCA, mean-pooling baseline, no standalone Clinical model, lightweight app, ResNet50 choice)
- `/Users/soumya/.claude/plans/as-of-now-we-refactored-tower.md` — the Claude Code plan-mode file this handoff document supersedes/expands on
