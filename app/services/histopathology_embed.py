"""Extract 2048-D histopathology embeddings from whole-slide (.svs) uploads."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "EXP"


def histopathology_inference_available() -> bool:
    """True when torch, openslide, and opencv are installed."""
    try:
        import cv2  # noqa: F401
        import openslide  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load_pipeline():
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from histopathology_utils import build_resnet50_encoder, embed_tiles, tile_slide

    return tile_slide, build_resnet50_encoder, embed_tiles


def svs_to_embedding_dataframe(svs_path: str | Path, patient_id: str) -> pd.DataFrame:
    """Tile an SVS slide, embed with ResNet50, mean-pool to one 2048-D vector."""
    if not histopathology_inference_available():
        raise RuntimeError(
            "Histopathology slide processing requires optional deps. "
            "Run: uv sync --extra app --extra histopathology"
        )

    tile_slide, build_resnet50_encoder, embed_tiles = _load_pipeline()
    tiles = tile_slide(str(svs_path))
    if not tiles:
        raise ValueError("No tissue tiles could be extracted from this slide.")

    encoder, device = build_resnet50_encoder()
    tile_embeddings = embed_tiles(encoder, device, tiles)
    patient_vec = tile_embeddings.mean(axis=0)

    row = {f"embed_{i}": float(patient_vec[i]) for i in range(patient_vec.shape[0])}
    df = pd.DataFrame([row], index=[patient_id])
    df.index.name = "PATIENT_ID"
    return df


def embedding_from_upload(uploaded_file, patient_id: str) -> pd.DataFrame:
    """Build a histopathology feature row from an uploaded .svs or precomputed .csv."""
    name = (uploaded_file.name or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        id_col = next(
            (c for c in ("PATIENT_ID", "patient_id", "Patient_ID") if c in df.columns),
            None,
        )
        if id_col:
            df = df.set_index(id_col)
        elif df.shape[0] == 1:
            df.index = [patient_id]
        df.index.name = "PATIENT_ID"
        return df

    if name.endswith(".svs"):
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            return svs_to_embedding_dataframe(tmp_path, patient_id)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    raise ValueError("Histopathology upload must be a .svs slide or precomputed .csv embedding.")
