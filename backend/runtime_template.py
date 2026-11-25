"""Chạy model đã xuất bằng NumPy thuần (không cần sklearn / TensorFlow).

Dùng:
    from gesture_runtime import GesturePredictor
    pred = GesturePredictor('xxx-portable.json')
    label, conf = pred.update(landmarks)   # landmarks: 21 điểm (x, y, z) chuẩn hoá 0..1 của MediaPipe
Hand 'both': pred.update(left_landmarks, right_landmarks).
Với model chuỗi (LSTM / KNN combo) gọi update() mỗi khung; trả ('unknown', 0.0) tới khi đủ khung.
Để landmark khớp lúc train, hãy chạy MediaPipe ở đúng `captureProfile` ghi trong manifest.
"""

import json
from typing import List, Optional, Sequence, Tuple

import numpy as np

from features import extract_features


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _act(name: str, x):
    if name == 'relu':
        return np.maximum(x, 0)
    if name in ('sigmoid', 'logistic'):
        return _sigmoid(x)
    if name == 'tanh':
        return np.tanh(x)
    if name == 'softmax':
        return _softmax(x)
    return x


class GesturePredictor:
    def __init__(self, portable_json_path: str) -> None:
        with open(portable_json_path, 'r', encoding='utf-8') as f:
            self.spec = json.load(f)
        self.kind = self.spec['kind']
        self.seq_frames = int(self.spec.get('sequenceFrames') or 0)
        self.hand_mode = self.spec.get('handMode', 'right')
        self.buffer: List[List[float]] = []

    def update(self, landmarks: Optional[Sequence], other: Optional[Sequence] = None) -> Tuple[str, float]:
        if self.hand_mode == 'both':
            left, right = landmarks, other
        elif self.hand_mode == 'left':
            left, right = landmarks, None
        else:
            left, right = None, landmarks
        lv = extract_features(list(left)) if left else None
        rv = extract_features(list(right)) if right else None
        if self.hand_mode == 'left':
            vec = lv
        elif self.hand_mode == 'right':
            vec = rv
        else:
            vec = None if lv is None or rv is None else [*lv, *rv]
        if vec is None:
            self.buffer.clear()
            return 'unknown', 0.0
        return self.predict_vec(vec)

    def reset(self) -> None:
        self.buffer.clear()

    def predict_vec(self, vec: List[float]) -> Tuple[str, float]:
        if self.kind == 'knn':
            return self._knn(vec)
        if self.kind == 'mlp':
            return self._mlp(np.asarray(vec, dtype=np.float32))
        return self._lstm(vec)

    def _window(self, vec: List[float]) -> Optional[np.ndarray]:
        self.buffer.append(vec)
        if len(self.buffer) > self.seq_frames:
            self.buffer = self.buffer[-self.seq_frames:]
        if len(self.buffer) < self.seq_frames:
            return None
        return np.asarray(self.buffer, dtype=np.float32)

    def _knn(self, vec):
        x = np.asarray(vec, dtype=np.float32)
        if self.seq_frames:
            w = self._window(vec)
            if w is None:
                return 'unknown', 0.0
            x = w.reshape(-1)
        X = np.asarray(self.spec['samples'], dtype=np.float32)
        d = np.linalg.norm(X - x, axis=1)
        idx = np.argsort(d)[: int(self.spec['k'])]
        votes = {}
        for i in idx:
            lab = self.spec['labels'][i]
            votes[lab] = votes.get(lab, 0) + 1
        return max(votes, key=votes.get), 1.0

    def _mlp(self, x):
        a = x
        n = len(self.spec['coefs'])
        for i in range(n):
            a = a @ np.asarray(self.spec['coefs'][i], dtype=np.float32) + np.asarray(self.spec['intercepts'][i], dtype=np.float32)
            if i < n - 1:
                a = _act(self.spec['activation'], a)
        classes = self.spec['classes']
        if self.spec['outActivation'] == 'logistic' and a.shape[-1] == 1:
            p1 = float(_sigmoid(a)[0])
            probs = np.asarray([1 - p1, p1])
        else:
            probs = _softmax(a)
        i = int(np.argmax(probs))
        return str(classes[i]), float(probs[i])

    def _lstm(self, vec):
        w = self._window(vec)
        if w is None:
            return 'unknown', 0.0
        if self.spec['impl'] == 'mlp-flat':
            return self._mlp(w.reshape(-1))
        h = None
        for layer in self.spec['layers']:
            if layer['type'] == 'lstm':
                K, R, b = (np.asarray(m, dtype=np.float32) for m in layer['weights'])
                u = R.shape[0]
                h = np.zeros(u, dtype=np.float32)
                c = np.zeros(u, dtype=np.float32)
                for t in range(w.shape[0]):
                    z = w[t] @ K + h @ R + b
                    i_, f_, g_, o_ = z[:u], z[u:2 * u], z[2 * u:3 * u], z[3 * u:]
                    c = _sigmoid(f_) * c + _sigmoid(i_) * np.tanh(g_)
                    h = _sigmoid(o_) * np.tanh(c)
            else:
                W, b = (np.asarray(m, dtype=np.float32) for m in layer['weights'])
                h = _act(layer['activation'], h @ W + b)
        labels = self.spec['labels']
        if h.shape[-1] == 1:
            return (labels[0] if labels else 'gesture'), float(h[0])
        i = int(np.argmax(h))
        return str(labels[i]), float(h[i])
