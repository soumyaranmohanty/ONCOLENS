"""Histopathology WSI preprocessing and ResNet50 feature extraction for ONCOLENS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def macenko_normalize(img_rgb: np.ndarray, alpha: float = 1.0, beta: float = 0.15) -> np.ndarray:
    """Macenko stain normalization on an RGB uint8 image (H, W, 3)."""
    img = img_rgb.astype(np.float32)
    img = np.maximum(img, 1.0)
    od = -np.log(img / 255.0)

    mask = (img[:, :, 0] > 220) & (img[:, :, 1] > 220) & (img[:, :, 2] > 220)
    od_flat = od[~mask].reshape(-1, 3)
    if od_flat.shape[0] < 1000:
        od_flat = od.reshape(-1, 3)

    _, _, v = np.linalg.svd(od_flat, full_matrices=False)
    v = v[:2, :]
    proj = od_flat @ v.T
    phi = np.arctan2(proj[:, 1], proj[:, 0])

    min_phi, max_phi = np.percentile(phi, (100 * beta, 100 * (1 - beta)))
    v1 = np.array([np.cos(min_phi), np.sin(min_phi)]) @ v
    v2 = np.array([np.cos(max_phi), np.sin(max_phi)]) @ v

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    stain = np.array([v1, v2])
    concentrations = od.reshape(-1, 3) @ stain.T
    concentrations = np.maximum(concentrations, 1e-6)

    stain_ref = np.array([[0.644, 0.717, 0.267], [0.093, 0.954, 0.283]])
    od_norm = concentrations @ stain_ref
    img_norm = np.exp(-od_norm * alpha) * 255.0
    img_norm = img_norm.reshape(img_rgb.shape)
    return np.clip(img_norm, 0, 255).astype(np.uint8)


def _tissue_mask_otsu(thumbnail_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask > 0


def tile_slide(
    svs_path: str,
    tile_size: int = 256,
    magnification: int = 20,
    max_background_fraction: float = 0.85,
    tissue_thumbnail_max_dim: int = 2048,
    max_tiles: int = 500,
) -> list[tuple[str, Image.Image]]:
    """Extract tissue tiles at target magnification with Macenko normalization."""
    slide = openslide.OpenSlide(svs_path)
    try:
        objective = float(slide.properties.get("openslide.objective-power", magnification))
        downsample = max(objective / magnification, 1.0)
        level_dims = slide.level_dimensions[0]
        read_level = slide.get_best_level_for_downsample(downsample)
        level_downsample = slide.level_downsamples[read_level]
        tile_stride = int(tile_size * level_downsample)

        thumb_w, thumb_h = slide.level_dimensions[0]
        scale = min(1.0, tissue_thumbnail_max_dim / max(thumb_w, thumb_h))
        thumb_size = (max(1, int(thumb_w * scale)), max(1, int(thumb_h * scale)))
        thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
        tissue_mask = _tissue_mask_otsu(np.array(thumbnail))

        mask_scale_x = tissue_mask.shape[1] / level_dims[0]
        mask_scale_y = tissue_mask.shape[0] / level_dims[1]

        ys, xs = np.where(tissue_mask)
        if len(xs) == 0:
            return []
        x0 = int(xs.min() / mask_scale_x)
        y0 = int(ys.min() / mask_scale_y)
        x1 = int(np.ceil(xs.max() / mask_scale_x))
        y1 = int(np.ceil(ys.max() / mask_scale_y))

        coords: list[tuple[int, int]] = []
        for y in range(y0, min(y1, level_dims[1] - tile_stride) + 1, tile_stride):
            for x in range(x0, min(x1, level_dims[0] - tile_stride) + 1, tile_stride):
                mx0 = int(x * mask_scale_x)
                my0 = int(y * mask_scale_y)
                mx1 = int((x + tile_stride) * mask_scale_x)
                my1 = int((y + tile_stride) * mask_scale_y)
                region = tissue_mask[my0:my1, mx0:mx1]
                if region.size == 0:
                    continue
                if region.mean() < (1.0 - max_background_fraction):
                    continue
                coords.append((x, y))

        if len(coords) > max_tiles:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(coords), size=max_tiles, replace=False)
            coords = [coords[i] for i in sorted(idx)]

        tiles: list[tuple[str, Image.Image]] = []
        for tile_idx, (x, y) in enumerate(coords):
            patch = slide.read_region((x, y), read_level, (tile_size, tile_size)).convert("RGB")
            arr = macenko_normalize(np.array(patch))
            tiles.append((f"tile_{tile_idx:05d}", Image.fromarray(arr)))
        return tiles
    finally:
        slide.close()


def build_resnet50_encoder() -> tuple[nn.Module, torch.device]:
    """Load frozen ImageNet ResNet50 encoder (2048-D output)."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    device = _get_device()
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, device


def _preprocess_transform() -> transforms.Compose:
    weights = ResNet50_Weights.IMAGENET1K_V2
    return weights.transforms()


def embed_tiles(
    encoder: nn.Module,
    device: torch.device,
    tiles: Iterable[tuple[str, Image.Image]],
    batch_size: int = 32,
) -> np.ndarray:
    """Run frozen ResNet50 on tiles; return (n_tiles, 2048) float32 array."""
    transform = _preprocess_transform()
    tile_list = list(tiles)
    if not tile_list:
        return np.zeros((0, 2048), dtype=np.float32)

    embeddings: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = encoder(batch).cpu().numpy()
        embeddings.append(out)
        batch_tensors.clear()

    for _, pil_img in tile_list:
        batch_tensors.append(transform(pil_img))
        if len(batch_tensors) >= batch_size:
            flush_batch()
    flush_batch()
    return np.vstack(embeddings).astype(np.float32)


def cache_slide_embeddings(
    slide_id: str,
    svs_path: str,
    cache_dir: str | Path,
    encoder: nn.Module | None = None,
    device: torch.device | None = None,
) -> Path:
    """Tile a slide, embed tiles, save (n_tiles, 2048) .npy cache. Returns cache path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{slide_id}.npy"
    if cache_path.exists():
        return cache_path

    tiles = tile_slide(svs_path)
    if encoder is None or device is None:
        encoder, device = build_resnet50_encoder()
    emb = embed_tiles(encoder, device, tiles)
    np.save(cache_path, emb)
    return cache_path


def aggregate_patient_features(slide_manifest: pd.DataFrame, tile_cache_dir: str | Path) -> pd.DataFrame:
    """Mean-pool tile embeddings per slide, then per patient → 2048-D feature matrix."""
    tile_cache_dir = Path(tile_cache_dir)
    patient_slides: dict[str, list[np.ndarray]] = {}

    for _, row in slide_manifest.iterrows():
        slide_id = Path(row["slide_id"]).stem
        cache_path = tile_cache_dir / f"{slide_id}.npy"
        if not cache_path.exists():
            continue
        tile_emb = np.load(cache_path)
        if tile_emb.size == 0:
            continue
        slide_mean = tile_emb.mean(axis=0)
        patient_slides.setdefault(row["PATIENT_ID"], []).append(slide_mean)

    records = {}
    embed_dim = 2048
    for patient_id, slide_means in patient_slides.items():
        patient_vec = np.mean(slide_means, axis=0)
        records[patient_id] = {f"embed_{i}": float(patient_vec[i]) for i in range(embed_dim)}

    if not records:
        cols = [f"embed_{i}" for i in range(embed_dim)]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "PATIENT_ID"
    return df
