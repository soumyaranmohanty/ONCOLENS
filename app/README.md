# ONCOLENS Streamlit App

## Run

From the **repository root**:

```bash
uv sync --extra app
uv run streamlit run app/oncolens_inference.py
```

## Modules

1. **Dashboard** — project overview, model performance, architecture
2. **Patient Prediction** — demo TCGA patients or CSV upload; 5 models + compare all
3. **Model Analytics** — metrics, ROC, confusion matrix from test predictions
4. **AI Clinical Report** — rule-based report with PDF export
5. **Treatment Recommendation** — educational biomarker-driven suggestions
6. **Research & Explainability** — feature importance, mutation prevalence, AUC comparison
7. **Virtual AI Assistant** — FAQ and prediction explainer (rule-based)
8. **About** — project info and references

## Notes

- Histopathology accepts **precomputed** 2048-D embedding vectors only (no `.svs` upload).
- **4-Modality** predictions activate automatically when `Data/feature_store/multimodal_model_v4/` exists.
- Requires `Data/feature_store/` artifacts from the training pipeline.
