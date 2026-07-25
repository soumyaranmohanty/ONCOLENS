# Suggested Updates to the Histopathology Action Plan

## Overall Decision

The overall action plan is well aligned with the original project
proposal and can be followed with a few modifications. The primary
objective remains to extend the current genomics-based platform into a
four-modality precision oncology framework by incorporating
Histopathology.

## Change 1: Build and Validate the Histopathology Model First

Develop and evaluate the Histopathology module as a standalone modality
before integrating it into the multimodal framework.

Workflow:

Whole Slide Images → Preprocessing → Tile Extraction → ResNet50 Feature
Extraction → Patient-level Feature Aggregation → Standalone
Histopathology Model → Performance Evaluation → 4-Modality Fusion

Rationale: - Independent validation of the image pipeline. - Easier
debugging. - Scientific comparison of Histopathology as an individual
modality. - Cleaner reporting.

## Change 2: Make PCA Optional

Treat PCA as an optional optimization instead of a mandatory
preprocessing step.

Preferred baseline:

ResNet50 → 2048-D Image Embeddings → Random Forest / Logistic Regression

Only evaluate PCA if dimensionality or runtime becomes a concern.

## Change 3: Keep Mean Pooling as the Baseline

Tile Embeddings → Mean Pooling → Slide Embedding → Patient Embedding

Advanced aggregation methods (Attention Pooling, MIL, CLAM) can be
considered as future work.

## Change 4: Skip the Standalone Clinical Model

Decision: A standalone Clinical-only model will NOT be implemented.

Reason: The project has already completed: - Gene Expression Model -
Mutation Model - Multi-Modal Model (Clinical + Gene Expression +
Mutation)

The focus is to extend the existing multimodal framework with
Histopathology rather than introducing another baseline.

Remove: - Clinical-only notebook - registry_clinical.json -
Clinical-only comparisons

Final comparison:

-   Gene Expression
-   Mutation
-   Histopathology
-   Existing 3-Modality Model (Clinical + Gene Expression + Mutation)
-   New 4-Modality Model (Clinical + Gene Expression + Mutation +
    Histopathology)

## Change 5: Keep the Streamlit Application Lightweight

Do not implement live Whole Slide Image (.svs) inference.

If Histopathology is added to the application later, use precomputed
image feature vectors instead of raw slide uploads.

## Change 6: Continue with ImageNet-Pretrained ResNet50

Continue using ImageNet-pretrained ResNet50 as the baseline feature
extractor.

Reasons: - Stable baseline - Easy integration - Compatible with Apple
Silicon (MPS) - Appropriate for transfer learning - Aligns with the
project's objective

## Revised Execution Order

Phase 4 -- Histopathology Module

1.  Download TCGA-LUAD Whole Slide Images.
2.  Tissue detection and tile extraction.
3.  Stain normalization.
4.  ResNet50 feature extraction.
5.  Patient-level feature aggregation.
6.  Train and evaluate the Histopathology model.

Phase 5 -- Four-Modality Integration

1.  Merge Histopathology features with Clinical, Gene Expression and
    Mutation.
2.  Retrain the multimodal model.
3.  Compare the existing 3-modality model with the new 4-modality model.

Phase 6 -- Comparative Evaluation

Compare: - Gene Expression - Mutation - Histopathology - Existing
3-Modality Multi-Modal Model - New 4-Modality Multi-Modal Model

Metrics: - Accuracy - Precision - Recall - F1-Score - ROC-AUC

## Final Recommendation

The action plan is approved with the above modifications. These changes
align with the original project proposal while keeping the
implementation practical, focused, and consistent with the existing
architecture.
