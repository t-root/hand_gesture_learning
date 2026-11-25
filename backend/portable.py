"""Xuất model ra JSON thuần (số), để chạy ngoài app (NumPy / trình duyệt)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def mlp_spec(clf: Any) -> Dict[str, Any]:
    return {
        'coefs': [c.astype('float32').tolist() for c in clf.coefs_],
        'intercepts': [b.astype('float32').tolist() for b in clf.intercepts_],
        'activation': str(clf.activation),
        'outActivation': str(clf.out_activation_),
        'classes': [str(c) for c in clf.classes_],
    }


def keras_layers(model: Any) -> list:
    out = []
    for layer in model.layers:
        name = layer.__class__.__name__
        if name == 'LSTM':
            out.append({'type': 'lstm', 'weights': [w.astype('float32').tolist() for w in layer.get_weights()]})
        elif name == 'Dense':
            act = layer.get_config().get('activation', 'linear')
            out.append({'type': 'dense', 'activation': act, 'weights': [w.astype('float32').tolist() for w in layer.get_weights()]})
    return out


def build_portable(
    mode: str,
    *,
    backend: str,
    model: Any,
    label_map: list,
    knn_samples: Optional[tuple],
    knn_k: int,
    sequence: bool,
    sequence_frames: int,
    hand_mode: str,
) -> Optional[Dict[str, Any]]:
    base: Dict[str, Any] = {'handMode': hand_mode}
    if mode == 'knn':
        if not knn_samples or not knn_samples[0]:
            return None
        xs, ys = knn_samples
        return {**base, 'kind': 'knn', 'k': knn_k, 'samples': xs, 'labels': ys,
                'sequenceFrames': sequence_frames if sequence else 0}
    if model is None:
        return None
    if mode == 'mlp':
        return {**base, 'kind': 'mlp', **mlp_spec(model)}
    if backend == 'tensorflow':
        return {**base, 'kind': 'lstm', 'impl': 'keras', 'sequenceFrames': sequence_frames,
                'labels': list(label_map), 'layers': keras_layers(model)}
    return {**base, 'kind': 'lstm', 'impl': 'mlp-flat', 'sequenceFrames': sequence_frames, **mlp_spec(model)}
