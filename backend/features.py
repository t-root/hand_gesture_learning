"""Trích đặc trưng từ landmark và tiện ích số."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec='seconds') + 'Z'


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def angle(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> float:
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=np.float32)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc)) or 1.0
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.arccos(cosang))


def extract_features(landmarks: List[Tuple[float, float, float]]) -> List[float]:
    if not landmarks:
        return [0.0] * 71
    base = landmarks[0]
    ref = landmarks[9] if len(landmarks) > 9 else base
    scale = math.dist((ref[0], ref[1], ref[2]), (base[0], base[1], base[2])) or 1.0
    f: List[float] = []
    for x, y, z in landmarks:
        f.append((x - base[0]) / scale)
        f.append((y - base[1]) / scale)
        f.append((z - base[2]) / scale)
    idx_pairs = [
        (0, 5, 6), (5, 6, 8),
        (0, 9, 10), (9, 10, 12),
        (0, 13, 14), (13, 14, 16),
        (0, 17, 18), (17, 18, 20),
    ]
    for a, b, c in idx_pairs:
        if max(a, b, c) < len(landmarks):
            f.append(angle(landmarks[a], landmarks[b], landmarks[c]))
        else:
            f.append(0.0)
    return f


def zero_vec(length: int = 71) -> List[float]:
    return [0.0] * length


def combine_vec(a: List[float], b: List[float]) -> List[float]:
    return [*a, *b]
