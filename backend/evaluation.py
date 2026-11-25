"""Đánh giá model bằng tập kiểm tra theo thời gian (không xáo trộn).

Mỗi nhãn: 80% mẫu thu trước dùng để train, 20% mẫu thu sau cùng dùng để kiểm tra.
Chia theo thời gian tránh lỗi "rò rỉ": các khung liền kề gần như giống hệt nhau, nếu xáo trộn
ngẫu nhiên thì độ chính xác sẽ cao giả tạo. Nên thu mẫu kiểm tra ở một lượt riêng (khác tư thế,
khác ánh sáng) để số liệu sát thực tế.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from .deps import Dense, Dropout, LSTM, Sequential, tf

VAL_FRACTION = 0.2
MIN_PER_LABEL = 4


def split_by_time(y: List[str], frac: float = VAL_FRACTION) -> Tuple[List[int], List[int]]:
    by_label: Dict[str, List[int]] = {}
    for i, lab in enumerate(y):
        by_label.setdefault(str(lab), []).append(i)
    train: List[int] = []
    val: List[int] = []
    for idxs in by_label.values():
        n_val = max(1, int(round(len(idxs) * frac)))
        train.extend(idxs[:-n_val])
        val.extend(idxs[-n_val:])
    return sorted(train), sorted(val)


def _fit_predict(mode: str, Xtr: np.ndarray, ytr: List[str], Xva: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Trả (nhãn dự đoán, độ tin cậy) cho tập kiểm tra."""
    if mode == 'knn':
        clf = KNeighborsClassifier(n_neighbors=min(3, len(ytr)) or 1)
        clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
        pred = clf.predict(Xva.reshape(len(Xva), -1))
        return [str(p) for p in pred], np.ones(len(pred))
    if mode == 'lstm' and tf is not None and Sequential is not None and Xtr.ndim == 3:
        labels = sorted(set(ytr))
        idx = {lab: i for i, lab in enumerate(labels)}
        model = Sequential()
        model.add(LSTM(32, input_shape=(Xtr.shape[1], Xtr.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(labels), activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.fit(Xtr, np.asarray([idx[v] for v in ytr]), epochs=25, batch_size=min(16, len(Xtr)), shuffle=True, verbose=0)
        prob = np.asarray(model.predict(Xva, verbose=0))
        top = prob.argmax(axis=1)
        return [labels[i] for i in top], prob[np.arange(len(top)), top]
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)
    clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
    prob = clf.predict_proba(Xva.reshape(len(Xva), -1))
    top = prob.argmax(axis=1)
    return [str(clf.classes_[i]) for i in top], prob[np.arange(len(top)), top]


def evaluate_holdout(mode: str, X: List[Any], y: List[str], threshold: float) -> Dict[str, Any]:
    y = [str(v) for v in y]
    labels = sorted(set(y))
    counts = {lab: y.count(lab) for lab in labels}
    if len(labels) < 2:
        return {'ok': False, 'text': 'Need 2+ labels (add NONE).'}
    small = [lab for lab, c in counts.items() if c < MIN_PER_LABEL]
    if small:
        return {'ok': False, 'text': 'Need ' + str(MIN_PER_LABEL) + '+ samples per label: ' + ', '.join(small)}
    arr = np.asarray(X, dtype=np.float32)
    tr, va = split_by_time(y)
    pred, conf = _fit_predict(mode, arr[tr], [y[i] for i in tr], arr[va])
    truth = [y[i] for i in va]
    use_thr = mode != 'knn'
    final = [p if (not use_thr or c >= threshold) else 'unknown' for p, c in zip(pred, conf)]
    correct = sum(1 for t, p in zip(truth, final) if t == p)
    per: Dict[str, Dict[str, Any]] = {}
    for lab in labels:
        n_true = sum(1 for t in truth if t == lab)
        hit = sum(1 for t, p in zip(truth, final) if t == lab and p == lab)
        n_pred = sum(1 for p in final if p == lab)
        per[lab] = {
            'n': n_true,
            'recall': hit / n_true if n_true else None,
            'missed': n_true - hit,
            'falseAlarms': n_pred - hit,
            'precision': hit / n_pred if n_pred else None,
        }
    confusion: Dict[str, Dict[str, int]] = {t: {} for t in labels}
    for t, p in zip(truth, final):
        confusion[t][p] = confusion[t].get(p, 0) + 1
    acc = correct / len(truth)
    lines = [f'{mode.upper()} holdout: {correct}/{len(truth)} correct ({acc * 100:.0f}%), trained on {len(tr)}']
    for lab in labels:
        d = per[lab]
        rec = '-' if d['recall'] is None else f"{d['recall'] * 100:.0f}%"
        lines.append(f'  {lab}: caught {d["n"] - d["missed"]}/{d["n"]} ({rec}) • missed {d["missed"]} • false alarms {d["falseAlarms"]}')
    unk = sum(1 for p in final if p == 'unknown')
    if unk:
        lines.append(f'  below threshold {threshold:.2f}: {unk} counted as unknown')
    few = [lab for lab in labels if per[lab]['n'] < 10]
    if few:
        names = ', '.join(l + '=' + str(per[l]['n']) for l in few)
        lines.append('Too few test samples: ' + names)
    return {
        'ok': True, 'accuracy': acc, 'nTrain': len(tr), 'nVal': len(va), 'perLabel': per,
        'confusion': confusion, 'threshold': threshold if use_thr else None, 'text': '\n'.join(lines),
    }
