# Additional Recommended Updates to the Action Plan

These recommendations are intended as refinements to the current
implementation plan. The overall architecture and sequencing are already
well aligned with the project objectives.

------------------------------------------------------------------------

## 1. Explicitly Define the Objective of Phase 4

### Current Situation

Phase 4 immediately starts describing implementation tasks.

### Recommended Update

Add the following objective before Section 4.1:

> **Phase 4 Objective**
>
> Develop, train, validate, and evaluate the Histopathology prediction
> model as a standalone modality before integrating it into the
> four-modality multimodal framework.

### Why

This clearly establishes that Histopathology is an independent model
whose performance must first be validated before fusion.

------------------------------------------------------------------------

## 2. Clarify the Role of Histopathology in the Existing Architecture

Under **Section 4.4 (Repo Convention)**, add the following note:

> The Histopathology model should follow the same Level-0 modelling
> paradigm already established for the Gene Expression and Mutation
> models. The generated prediction probabilities become an additional
> Level-0 input to the existing stacking architecture.

### Why

This connects the new module with the current multimodal design and
makes the architectural evolution easier to understand.

------------------------------------------------------------------------

## 3. Rename the Comparative Evaluation Section

Instead of:

> Final comparison set

Use:

> **Final Experimental Comparison**

The comparison should include:

-   Gene Expression Model
-   Mutation Model
-   Histopathology Model
-   Existing 3-Modality Model (Clinical + Gene Expression + Mutation)
-   New 4-Modality Model (Clinical + Gene Expression + Mutation +
    Histopathology)

### Why

This terminology is more consistent with academic research papers and
thesis writing.

------------------------------------------------------------------------

## 4. Add Feature Contribution Analysis

Under **Phase 5 -- Verification**, add a new subsection.

### Feature Contribution Analysis

After training the 4-modality model, evaluate the contribution of the
Histopathology modality by comparing it against the existing 3-modality
model using the same patient cohort.

Report:

-   Absolute AUC Improvement
-   Relative Performance Improvement
-   Accuracy Difference
-   F1-Score Difference

### Why

This demonstrates the value added by incorporating Histopathology into
the multimodal framework and provides one of the key findings of the
project.

------------------------------------------------------------------------

## 5. Add a Final Deliverables Phase

Add a final section after Comparative Evaluation.

# Phase 7 -- Final Deliverables

Complete the following project artefacts:

-   Final Project Report
-   Architecture Diagrams
-   Performance Analysis
-   Comparative Evaluation Results
-   Final Presentation
-   Source Code Documentation
-   Git Repository Cleanup

### Why

This clearly defines the completion criteria for the project beyond
implementation and experimentation.

------------------------------------------------------------------------

## 6. Revise the Baseline Sanity Check

Current wording:

> Histopathology Level-0 CV/test AUC should clear \~0.5 meaningfully
> before moving to fusion.

Recommended wording:

> The standalone Histopathology model should demonstrate predictive
> performance above random guessing before proceeding to multimodal
> integration.

### Why

Avoid prescribing a fixed numerical threshold in the implementation
plan. The appropriate performance level depends on dataset
characteristics and class balance.

------------------------------------------------------------------------

# Summary

The current implementation plan is technically sound and closely aligned
with the original project proposal.

The above refinements improve:

-   Architectural clarity
-   Research methodology
-   Experimental design
-   Evaluation strategy
-   Final project organization

These changes do not alter the implementation itself but make the
roadmap stronger for documentation, thesis writing, and project
presentations.
