# -*- coding: utf-8 -*-
"""Metrics utilities for Smart Product Pricing."""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def smape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    val = float(np.mean(np.where(den > eps, num / den, 0.0)))
    logger.debug(f"SMAPE computed on n={y_true.shape[0]} -> {val:.6f}")
    return val

def smooth_smape_loss(y_true, y_pred, eps=1e-3):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    val = float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))
    logger.debug(f"Smooth-SMAPE computed on n={y_true.shape[0]} -> {val:.6f}")
    return val

def log_mae(y_true, y_pred):
    val = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    logger.debug(f"Log-MAE computed on n={len(y_true)} -> {val:.6f}")
    return val