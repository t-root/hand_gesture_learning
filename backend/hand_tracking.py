"""Tải mô hình MediaPipe Tasks và adapter kết quả tay (tương thích API cũ)."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .deps import mp
from .paths import STORAGE_DIR

HAND_LANDMARKER_MODEL = 'hand_landmarker.task'
HAND_LANDMARKER_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/latest/hand_landmarker.task'
)


def ensure_hand_landmarker_model() -> Optional[Path]:
    dest = STORAGE_DIR / HAND_LANDMARKER_MODEL
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    tmp = dest.with_suffix('.task.download')
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, tmp)
        tmp.replace(dest)
        return dest if dest.is_file() else None
    except Exception:
        try:
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return None


class _LegacyHandLandmark:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _LegacyHandLandmarks:
    def __init__(self, landmark: List[_LegacyHandLandmark]) -> None:
        self.landmark = landmark


class _LegacyHandClassification:
    def __init__(self, label: str) -> None:
        self.label = label


class _LegacyHandedness:
    def __init__(self, label: str) -> None:
        self.classification = [_LegacyHandClassification(label)]


class _LegacyHandResults:
    def __init__(
        self,
        multi_hand_landmarks: Optional[List[_LegacyHandLandmarks]],
        multi_handedness: Optional[List[_LegacyHandedness]],
    ) -> None:
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness


def task_hand_result_to_legacy(result: Any) -> _LegacyHandResults:
    if not result or not getattr(result, 'hand_landmarks', None):
        return _LegacyHandResults(None, None)
    multi: List[_LegacyHandLandmarks] = []
    handed: List[_LegacyHandedness] = []
    for i, lm_list in enumerate(result.hand_landmarks):
        pts = [
            _LegacyHandLandmark(
                float(lm.x if lm.x is not None else 0.0),
                float(lm.y if lm.y is not None else 0.0),
                float(lm.z if lm.z is not None else 0.0),
            )
            for lm in lm_list
        ]
        multi.append(_LegacyHandLandmarks(pts))
        label = 'Right'
        h = getattr(result, 'handedness', None) or []
        if i < len(h) and h[i]:
            c0 = h[i][0]
            label = str(c0.category_name or c0.display_name or 'Right')
        handed.append(_LegacyHandedness(label))
    return _LegacyHandResults(multi, handed)


class TaskHandLandmarkerAdapter:
    """HandLandmarker (VIDEO) với API giống `Hands.process`."""

    def __init__(self, landmarker: Any) -> None:
        self._landmarker = landmarker
        self._ts_ms = 0
        self._t0 = time.perf_counter()

    def process(self, img_rgb: np.ndarray) -> _LegacyHandResults:
        # Timestamp thực (ms) để tracking VIDEO khớp nhịp khung thật; luôn tăng dần.
        now_ms = int((time.perf_counter() - self._t0) * 1000)
        self._ts_ms = max(self._ts_ms + 1, now_ms)
        mp_image = mp.Image(mp.ImageFormat.SRGB, img_rgb)  # type: ignore[union-attr]
        raw = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        return task_hand_result_to_legacy(raw)

    def close(self) -> None:
        if self._landmarker is not None and hasattr(self._landmarker, 'close'):
            self._landmarker.close()
            self._landmarker = None
