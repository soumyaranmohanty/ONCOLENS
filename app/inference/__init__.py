"""Inference utilities for ONCOLENS predictions."""

from app.inference.predict import compare_all, predict_multimodal, predict_single

__all__ = ["predict_single", "predict_multimodal", "compare_all"]
