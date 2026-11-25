from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import re
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from .deps import Dense, Dropout, LSTM, MEDIAPIPE_TASKS, Sequential, mp, tf
from .features import combine_vec, extract_features, now_iso, safe_float, safe_int, zero_vec
from .hand_tracking import TaskHandLandmarkerAdapter, ensure_hand_landmarker_model
from .paths import APP_DIR, EXPORT_DIR, STORAGE_DIR
from .evaluation import evaluate_holdout
from .portable import build_portable

NEGATIVE_LABEL = 'none'  # lớp "không phải cử chỉ" (tay nghỉ / cử động khác)
DEFAULT_DETECT_WIDTH = 480
DEFAULT_MAX_FPS = 24


@dataclass
class ModeState:
    sample_count: int = 0
    sample_buffer_count: Dict[str, int] = field(default_factory=dict)
    labels: set = field(default_factory=set)
    tensor_x: List[Any] = field(default_factory=list)
    tensor_y: List[str] = field(default_factory=list)
    tensor_hand_modes: List[str] = field(default_factory=list)
    feature_length: int = 71
    label_map: List[str] = field(default_factory=list)
    model: Any = None
    trained: bool = False
    backend: str = 'sklearn'
    scaler: Any = None

    def clear(self) -> None:
        self.sample_count = 0
        self.sample_buffer_count.clear()
        self.labels.clear()
        self.tensor_x.clear()
        self.tensor_y.clear()
        self.tensor_hand_modes.clear()
        self.feature_length = 71
        self.label_map.clear()
        self.model = None
        self.trained = False
        self.backend = 'sklearn'
        self.scaler = None


class AppState:
    def __init__(self) -> None:
        self.active_model = 'knn'
        self.recognition_mode = 'single'
        self.hand_mode = 'right'
        self.target_count = 100
        self.stored_target_count = 100
        self.sequence_frames = 30
        self.threshold = 0.6
        self.detect_width = DEFAULT_DETECT_WIDTH
        self.max_fps = DEFAULT_MAX_FPS
        self.countdown_s = 2
        self.batch_total = 10
        self.batch_done = 0
        self.learn_base = 0
        self.eval_reports: Dict[str, Any] = {}
        self._evaluating = False
        self.learn_start_at = 0.0
        self.learning = False
        self.testing = False
        self.training = False
        self._train_cancel = False
        self._train_lock = threading.Lock()
        self.train_progress: Dict[str, Any] = {}
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_send: Optional[Any] = None
        self.running = False
        self.current_label = ''
        self.current_sample_progress = 0
        self.last_prediction_label = '-'
        self.last_prediction_confidence = 0.0
        self.prediction_text = '-'
        self.recent_committed_labels: List[str] = []
        self.stable_label = ''
        self.stable_count = 0
        self.current_sequence: List[List[float]] = []
        self.test_sequence: List[List[float]] = []
        self.sequence_rules: List[Dict[str, Any]] = []
        self.last_notice = 'Ready.'
        self.last_notice_tone = 'good'
        self.status_sticky = False
        self.last_frame_fps = 0.0
        self.last_frame_at = time.perf_counter()
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn_fitted = False
        self.modes = {
            'knn': ModeState(),
            'mlp': ModeState(),
            'lstm': ModeState(),
        }
        self.current_overlay: Dict[str, Any] = {'hands': []}
        self.current_status_by_mode = {'knn': 'KNN idle.', 'mlp': 'MLP idle.', 'lstm': 'LSTM idle.'}
        self.camera_info = 'CLOSED'
        self.conn_status = 'CAMERA OFF'
        self._hands_init_error = ''
        self._mp_hands = None
        if mp is not None:
            if MEDIAPIPE_TASKS:
                model_path = ensure_hand_landmarker_model()
                if model_path is None:
                    self._hands_init_error = (
                        'Could not obtain hand_landmarker.task (download failed). '
                        'Check network or place the file in the storage folder.'
                    )
                else:
                    try:
                        opts = mp.tasks.vision.HandLandmarkerOptions(
                            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                            running_mode=mp.tasks.vision.RunningMode.VIDEO,
                            num_hands=2,
                            min_hand_detection_confidence=0.6,
                            min_hand_presence_confidence=0.6,
                            min_tracking_confidence=0.6,
                        )
                        lm = mp.tasks.vision.HandLandmarker.create_from_options(opts)
                        self._mp_hands = TaskHandLandmarkerAdapter(lm)
                    except Exception as exc:
                        self._hands_init_error = f'MediaPipe HandLandmarker init failed: {exc}'
                        self._mp_hands = None
            elif hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                try:
                    self._mp_hands = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=0,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6,
                    )
                except Exception as exc:
                    self._hands_init_error = f'MediaPipe Hands (legacy) init failed: {exc}'
                    self._mp_hands = None
            else:
                self._hands_init_error = 'MediaPipe has no supported hand API (tasks or solutions).'

    def mode_state(self, mode: Optional[str] = None) -> ModeState:
        return self.modes[mode or self.active_model]

    def set_notice(self, msg: str, tone: str = 'good', *, sticky: bool = False) -> None:
        self.last_notice = msg
        self.last_notice_tone = tone
        if sticky:
            self.status_sticky = True

    def _apply_live_status_line(self) -> None:
        if self.status_sticky or self.learning:
            return
        self.last_notice_tone = 'warn' if not self.running else 'good'
        self.last_notice = self._program_status()

    def update_fps(self) -> None:
        now = time.perf_counter()
        dt = now - self.last_frame_at
        self.last_frame_at = now
        if dt > 0:
            inst = 1.0 / dt
            self.last_frame_fps = self.last_frame_fps * 0.9 + inst * 0.1 if self.last_frame_fps else inst

    def update_config(self, payload: Dict[str, Any]) -> None:
        active_model = payload.get('activeModel')
        if active_model in self.modes:
            self.active_model = active_model
        recognition_mode = payload.get('recognitionMode')
        if recognition_mode in ('single', 'combo'):
            self.recognition_mode = recognition_mode
        hand_mode = payload.get('handMode')
        if hand_mode in ('right', 'left', 'both'):
            self.hand_mode = hand_mode
        self.sequence_frames = max(5, safe_int(payload.get('sequenceFrames', self.sequence_frames), self.sequence_frames))
        self.threshold = min(1.0, max(0.0, safe_float(payload.get('threshold', self.threshold), self.threshold)))
        self.target_count = max(1, safe_int(payload.get('targetCount', self.target_count), self.target_count))
        self.stored_target_count = self.target_count
        if 'detectWidth' in payload:
            self.detect_width = min(1280, max(160, safe_int(payload.get('detectWidth'), self.detect_width)))
        if 'maxFps' in payload:
            self.max_fps = min(60, max(1, safe_int(payload.get('maxFps'), self.max_fps)))
        if 'countdown' in payload:
            self.countdown_s = min(10, max(0, safe_int(payload.get('countdown'), self.countdown_s)))
        if 'batchCount' in payload:
            self.batch_total = min(200, max(1, safe_int(payload.get('batchCount'), self.batch_total)))
        self.sequence_rules = []

    def capture_profile(self) -> Dict[str, Any]:
        """Điều kiện thu landmark; phải giống nhau giữa lúc thu/train và lúc chạy thật."""
        return {
            'detectWidth': self.detect_width,
            'maxFps': self.max_fps,
            'landmarker': 'mediapipe hand_landmarker float16',
            'numHands': 2,
            'minConfidence': 0.6,
        }

    def get_frame_vector(self, detected: Dict[str, Optional[List[float]]]) -> Optional[List[float]]:
        left = detected.get('Left')
        right = detected.get('Right')
        if self.hand_mode == 'left':
            return left
        if self.hand_mode == 'right':
            return right
        if self.hand_mode == 'both':
            if left is None or right is None:
                return None
            return combine_vec(left, right)
        return None

    def _is_combo_knn_sequence_mode(self) -> bool:
        return self.active_model in ('knn', 'lstm') and self.recognition_mode == 'combo'

    def _combo_sequence_target(self) -> int:
        # Combo: một record = đúng `sequence_frames` khung (không còn dùng chung với Sample target).
        return max(1, self.sequence_frames)

    def _label_count(self, label: str) -> int:
        mode = 'knn' if self.active_model == 'knn' else self.active_model
        return self.mode_state(mode).sample_buffer_count.get(label, 0)

    def record_negative(self) -> None:
        """Thu lớp NONE: tay nghỉ / cử động khác để model biết đâu là KHÔNG phải cử chỉ."""
        self.append_learning(NEGATIVE_LABEL)

    def _uses_sequence(self) -> bool:
        return self.active_model == 'lstm' or self._is_combo_knn_sequence_mode()

    @staticmethod
    def _aggregate_sequence(seq: List[List[float]]) -> Optional[List[float]]:
        if not seq:
            return None
        arr = np.asarray(seq, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        # Keep the full temporal order instead of averaging frames.
        # This makes Combo (KNN) behave more like a sequence capture model.
        return arr.reshape(-1).astype(np.float32).tolist()
    def _hand_mode_miss_notice(self) -> str:
        if self.hand_mode == 'left':
            return 'Cần tay TRÁI trong khung (Hand: Left).'
        if self.hand_mode == 'right':
            return 'Cần tay PHẢI trong khung (Hand: Right).'
        return 'Cần đủ HAI TAY trong khung (Hand: Both).'

    @staticmethod
    def _ensure_tensor_hand_modes_aligned(st: ModeState, fallback: str = 'right') -> None:
        fb = fallback if fallback in ('left', 'right', 'both') else 'right'
        n = len(st.tensor_y)
        hm = st.tensor_hand_modes
        if len(hm) < n:
            st.tensor_hand_modes = list(hm) + [fb] * (n - len(hm))
        elif len(hm) > n:
            st.tensor_hand_modes = hm[:n]
        for i in range(n):
            if st.tensor_hand_modes[i] not in ('left', 'right', 'both'):
                st.tensor_hand_modes[i] = fb

    def _sample_buffer_count_by_hand(self, st: ModeState) -> Dict[str, Dict[str, int]]:
        """Per label, only sides with count > 0 (no left/right/both zero padding)."""
        self._ensure_tensor_hand_modes_aligned(st, self.hand_mode)
        raw: Dict[str, Dict[str, int]] = {}
        for y, side in zip(st.tensor_y, st.tensor_hand_modes):
            if side not in ('left', 'right', 'both'):
                continue
            d = raw.setdefault(str(y), {})
            d[side] = d.get(side, 0) + 1
        out: Dict[str, Dict[str, int]] = {}
        for label, d in raw.items():
            slim = {k: v for k, v in d.items() if v > 0}
            if slim:
                out[label] = slim
        return out

    def get_feature_length(self) -> int:
        return 142 if self.hand_mode == 'both' else 71


    def _knn_expected_vector_length(self) -> int:
        base = self.get_feature_length()
        return base * self._combo_sequence_target() if self._is_combo_knn_sequence_mode() else base

    def _knn_companion_samples(self) -> Tuple[List[List[float]], List[str]]:
        expected_len = self._knn_expected_vector_length()
        xs: List[List[float]] = []
        ys: List[str] = []
        for vec, label in zip(self.mode_state('knn').tensor_x, self.mode_state('knn').tensor_y):
            try:
                if len(vec) != expected_len:
                    continue
            except Exception:
                continue
            xs.append(vec)
            ys.append(label)
        return xs, ys

    def _knn_model_matches_expected_length(self) -> bool:
        if not self.knn_fitted:
            return False
        fit_x = getattr(self.knn, '_fit_X', None)
        if fit_x is None:
            return True
        try:
            return int(fit_x.shape[1]) == self._knn_expected_vector_length()
        except Exception:
            return False

    def commit_label(self, label: str) -> None:
        if not label or label in ('unknown', NEGATIVE_LABEL):
            return
        if not self.recent_committed_labels or self.recent_committed_labels[-1] != label:
            self.recent_committed_labels.append(label)
            if len(self.recent_committed_labels) > 12:
                self.recent_committed_labels = self.recent_committed_labels[-12:]

    def process_prediction(self, raw_label: str, confidence: float) -> Tuple[str, float]:
        visible = raw_label if confidence >= self.threshold else 'unknown'
        if visible == 'unknown':
            self.stable_label = ''
            self.stable_count = 0
            self.last_prediction_label = 'unknown'
            self.last_prediction_confidence = confidence
            self.prediction_text = 'unknown'
            return 'unknown', confidence
        if visible == self.stable_label:
            self.stable_count += 1
        else:
            self.stable_label = visible
            self.stable_count = 1
        final_label = visible
        required_stable = 4
        if self.stable_count >= required_stable:
            self.commit_label(visible)
            self.stable_label = ''
            self.stable_count = 0
        self.last_prediction_label = final_label
        self.last_prediction_confidence = confidence
        self.prediction_text = final_label
        return final_label, confidence
    def build_knn_status(self) -> str:
        if self.knn_fitted and self._knn_model_matches_expected_length():
            return f'KNN trained • {len(self.mode_state("knn").labels)} label(s)'
        if self.knn_fitted:
            return 'KNN trained • incompatible with current mode/window'
        return 'KNN idle.'

    def build_mode_status(self, mode: str) -> str:
        st = self.mode_state(mode)
        if mode == 'knn':
            if not st.tensor_y:
                return 'KNN idle.'
            if self._is_combo_knn_sequence_mode():
                return f'Sequence KNN samples: {st.sample_count} • labels: {len(st.labels)} • frame target: {self._combo_sequence_target()}'
            return f'KNN samples: {st.sample_count} • labels: {len(st.labels)}'
        if mode == 'mlp':
            return 'MLP trained' if st.trained else ('MLP samples: %d • labels: %d' % (st.sample_count, len(st.labels)) if st.tensor_y else 'MLP idle.')
        if mode == 'lstm':
            if self._is_combo_knn_sequence_mode():
                return f'Sequence LSTM samples: {st.sample_count} • labels: {len(st.labels)} • frame target: {self._combo_sequence_target()}'
            return 'LSTM trained' if st.trained else ('LSTM samples: %d • labels: %d' % (st.sample_count, len(st.labels)) if st.tensor_y else 'LSTM idle.')
        return 'idle'
    def _snapshot_current_sample_progress(self) -> int:
        if self.active_model == 'lstm' or self._is_combo_knn_sequence_mode():
            if self.learning:
                return len(self.current_sequence)
            return self.current_sample_progress
        if self.active_model in ('knn', 'mlp') and self.current_label:
            st = self.mode_state(self.active_model)
            return int(st.sample_buffer_count.get(self.current_label, 0))
        return self.current_sample_progress
    def snapshot(self) -> Dict[str, Any]:
        st = self.mode_state()
        has_knn = self.knn_fitted and bool(self.mode_state('knn').labels)
        has_mlp = bool(self.mode_state('mlp').trained and self.mode_state('mlp').model)
        has_lstm = bool(self.mode_state('lstm').trained and self.mode_state('lstm').model)
        return {
            'activeModel': self.active_model,
            'recognitionMode': self.recognition_mode,
            'handMode': self.hand_mode,
            'running': self.running,
            'learning': self.learning,
            'testing': self.testing,
            'training': self.training,
            'trainProgress': dict(self.train_progress) if self.training and self.train_progress else None,
            'currentLabel': self.current_label,
            'targetCount': self.target_count,
            'storedTargetCount': self.stored_target_count,
            'sequenceFrames': self.sequence_frames,
            'threshold': self.threshold,
            'capture': {'detectWidth': self.detect_width, 'maxFps': self.max_fps},
            'countdown': self.countdown_s,
            'batch': {'total': self.batch_total, 'done': self.batch_done},
            'evalReport': (self.eval_reports.get(self.active_model) or {}).get('text'),
            'evaluating': self._evaluating,
            'hasNegative': NEGATIVE_LABEL in self.mode_state().labels,
            'currentSampleProgress': self._snapshot_current_sample_progress(),
            'sampleFps': round(self.last_frame_fps, 1),
            'predictionLabel': self.last_prediction_label,
            'predictionConfidence': self.last_prediction_confidence,
            'predictionText': self.prediction_text,
            'programStatus': self.last_notice,
            'programStatusTone': self.last_notice_tone,
            'live': {
                'mode': f'{self.active_model.upper()} {"ON" if self.running else "OFF"}',
                'camera': self.camera_info,
                'task': (
                    'TRAINING'
                    if self.training
                    else ('LEARNING' if self.learning else ('TESTING' if self.testing else 'IDLE'))
                ),
                'label': self.current_label or '-',
                'prediction': self.prediction_text if self.prediction_text != '-' else '-',
                'handMode': self.hand_mode.upper(),
            },
            'modeStatus': {
                'knn': self.current_status_by_mode['knn'],
                'mlp': self.current_status_by_mode['mlp'],
                'lstm': self.current_status_by_mode['lstm'],
            },
            'modeData': {
                'knn': self._mode_data_snapshot('knn', has_knn),
                'mlp': self._mode_data_snapshot('mlp', has_mlp),
                'lstm': self._mode_data_snapshot('lstm', has_lstm),
            },
            'overlay': self.current_overlay,
            'featureList': [
                {'name': 'Camera', 'value': 'ON' if self.running else 'OFF'},
                {'name': 'Type', 'value': 'SEQUENCE' if self.recognition_mode == 'combo' else 'STATIC'},
                {'name': 'Model', 'value': self.active_model.upper()},
                {'name': 'Hand', 'value': self.hand_mode.upper()},
                {'name': 'Record', 'value': f'YES ({self.current_label or "-"})' if self.learning else 'NO'},
                {'name': 'Test', 'value': 'YES' if self.testing else 'NO'},
                {'name': 'State', 'value': self._state_text()},
                {'name': 'Frames', 'value': self._snapshot_frames_feature()},
                {'name': 'Sequence', 'value': 'ON' if self.recognition_mode == 'combo' else 'OFF'},
            ],
            'hasModels': {'knn': has_knn, 'mlp': has_mlp, 'lstm': has_lstm},
        }

    def _state_text(self) -> str:
        st = self.mode_state()
        if self.active_model == 'knn':
            return 'KNN FAST' if self.knn_fitted else 'KNN EMPTY'
        if self.active_model == 'mlp':
            return 'MLP READY' if st.trained else 'MLP EMPTY'
        return 'LSTM READY' if st.trained else 'LSTM EMPTY'

    def _snapshot_frames_feature(self) -> str:
        if self.active_model == 'lstm' or self._is_combo_knn_sequence_mode():
            mode = 'lstm' if self.active_model == 'lstm' else 'knn'
            st = self.mode_state(mode)
            label = self.current_label or ''
            seq_done = st.sample_buffer_count.get(label, 0) if label else 0
            frame_prog = len(self.current_sequence)
            return f'{seq_done}/{self.target_count} seq • {frame_prog}/{self.sequence_frames} frm'
        return f'{self._snapshot_current_sample_progress()}/{self.target_count}'

    def _program_status(self) -> str:
        if self.learning:
            if self.active_model == 'lstm':
                st = self.mode_state('lstm')
                seq_done = st.sample_buffer_count.get(self.current_label, 0) if self.current_label else 0
                frame_prog = len(self.current_sequence)
                return (
                    f'RECORDING LSTM • {self.current_label or "-"} '
                    f'sequences {seq_done}/{self.target_count} • frames {frame_prog}/{self.sequence_frames}'
                )
            st = self.mode_state(self.active_model)
            cur = st.sample_buffer_count.get(self.current_label, 0) if self.current_label else 0
            return f'LEARNING {self.active_model.upper()} • {self.current_label or "-"} {cur}/{self.target_count}'
        if self.testing:
            return f'PROCESSING {self.active_model.upper()} • LIVE TEST'
        if not self.running:
            return 'CAMERA OFF'
        return f'{self.active_model.upper()} READY' if self.is_model_ready() else f'{self.active_model.upper()} READY • NO MODEL'

    def is_model_ready(self) -> bool:
        if self.active_model == 'knn':
            return self.knn_fitted and bool(self.mode_state('knn').labels)
        st = self.mode_state()
        return bool(st.trained and st.model)

    def _mode_data_snapshot(self, mode: str, has_model: bool) -> Dict[str, Any]:
        st = self.mode_state(mode)
        return {
            'sampleCount': st.sample_count,
            'sampleBufferCount': dict(st.sample_buffer_count),
            'sampleBufferCountByHand': self._sample_buffer_count_by_hand(st),
            'labels': sorted(list(st.labels)),
            'trained': st.trained,
            'featureLength': st.feature_length,
            'labelMap': list(st.label_map),
            'modelReady': has_model,
        }

    def _decode_image(self, data_url: str) -> Optional[np.ndarray]:
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        try:
            raw = base64.b64decode(data_url)
        except Exception:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w != self.detect_width:
            scale = self.detect_width / float(w)
            nh = max(1, int(round(h * scale)))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (self.detect_width, nh), interpolation=interp)
        return img

    def _detect_hands(self, img_bgr: np.ndarray) -> Tuple[Dict[str, Optional[List[float]]], Dict[str, Any]]:
        if self._mp_hands is None:
            return {'Left': None, 'Right': None}, {'hands': [], 'presentSlots': {'left': False, 'right': False}}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self._mp_hands.process(img_rgb)
        detected: Dict[str, Optional[List[float]]] = {'Left': None, 'Right': None}
        overlay_hands: List[Dict[str, Any]] = []
        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = str(handedness_list[i].classification[0].label if i < len(handedness_list) and handedness_list[i].classification else 'Right')
                label_l = label.lower()
                pts = [(float(lm.x), float(lm.y), float(lm.z if lm.z is not None else 0.0)) for lm in hand_landmarks.landmark]
                vec = extract_features(pts)
                if 'left' in label_l and detected['Left'] is None:
                    detected['Left'] = vec
                elif 'right' in label_l and detected['Right'] is None:
                    detected['Right'] = vec
                elif detected['Right'] is None:
                    detected['Right'] = vec
                elif detected['Left'] is None:
                    detected['Left'] = vec
                overlay_hands.append({'label': label, 'landmarks': pts})
        overlay: Dict[str, Any] = {
            'hands': overlay_hands,
            'width': int(img_bgr.shape[1]),
            'height': int(img_bgr.shape[0]),
            'presentSlots': {'left': detected['Left'] is not None, 'right': detected['Right'] is not None},
        }
        return detected, overlay

    def _knn_fit(self) -> None:
        st = self.mode_state('knn')
        if not st.tensor_x:
            self.knn_fitted = False
            return
        xs, ys = self._knn_companion_samples()
        if not xs:
            self.knn_fitted = False
            return
        X = np.asarray(xs, dtype=np.float32)
        y = np.asarray(ys)
        n_neighbors = min(3, len(ys)) or 1
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.knn.fit(X, y)
        self.knn_fitted = True

    def attach_websocket(
        self, loop: asyncio.AbstractEventLoop, send_coro: Any
    ) -> None:
        self._ws_loop = loop
        self._ws_send = send_coro

    def detach_websocket(self) -> None:
        self._ws_loop = None
        self._ws_send = None

    def _emit_ws_event(self, payload: Dict[str, Any]) -> None:
        loop = self._ws_loop
        send = self._ws_send
        if loop is None or send is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(send(payload), loop)
        except Exception:
            pass

    def _report_train_progress(self, mode: str, **kwargs: Any) -> None:
        total = int(kwargs.get('total_steps') or kwargs.get('total_epochs') or 1)
        step = int(kwargs.get('step') or kwargs.get('epoch') or 0)
        loss = kwargs.get('loss')
        accuracy = kwargs.get('accuracy')
        message = str(kwargs.get('message') or f'Training {mode.upper()} {step}/{total}')
        prog: Dict[str, Any] = {
            'mode': mode,
            'phase': str(kwargs.get('phase') or 'fit'),
            'epoch': int(kwargs.get('epoch') or step),
            'totalEpochs': int(kwargs.get('total_epochs') or total),
            'step': step,
            'totalSteps': total,
            'message': message,
        }
        if loss is not None:
            prog['loss'] = float(loss)
        if accuracy is not None:
            prog['accuracy'] = float(accuracy)
        self.train_progress = prog
        self.set_notice(message, 'good')
        self._emit_ws_event(
            {'type': 'train_progress', 'progress': prog, 'snapshot': self.snapshot()}
        )

    def _fit_sklearn_mlp_streaming(self, mode: str, X: np.ndarray, y: np.ndarray, total_iters: int) -> Optional[MLPClassifier]:
        chunk = 25
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            warm_start=True,
            max_iter=chunk,
            random_state=42,
        )
        self._report_train_progress(
            mode,
            phase='start',
            step=0,
            total_steps=total_iters,
            message=f'{mode.upper()} training 0/{total_iters}',
        )
        done = 0
        while done < total_iters:
            if self._train_cancelled():
                return None
            clf.max_iter = min(chunk, total_iters - done)
            clf.fit(X, y)
            done += clf.max_iter
            self._report_train_progress(
                mode,
                phase='fit',
                step=done,
                total_steps=total_iters,
                message=f'{mode.upper()} training {done}/{total_iters}',
            )
        return clf

    def _train_sklearn_mlp(self, st: ModeState) -> None:
        if self._train_cancelled():
            return
        X = np.asarray(st.tensor_x, dtype=np.float32)
        y = np.asarray(st.tensor_y)
        st.feature_length = int(X.shape[1])
        st.label_map = sorted(set(st.tensor_y))
        clf = self._fit_sklearn_mlp_streaming('mlp', X, y, 600)
        if clf is None or self._train_cancelled():
            return
        st.model = clf
        st.trained = True
        st.backend = 'sklearn'

    def _train_sklearn_lstm(self, st: ModeState) -> None:
        if self._train_cancelled():
            return
        X = np.asarray(st.tensor_x, dtype=np.float32)
        y = np.asarray(st.tensor_y)
        if X.ndim == 3:
            st.feature_length = int(X.shape[-1])
            X = X.reshape((X.shape[0], -1))
        st.label_map = sorted(set(st.tensor_y))
        clf = self._fit_sklearn_mlp_streaming('lstm', X, y, 700)
        if clf is None or self._train_cancelled():
            return
        st.model = clf
        st.trained = True
        st.backend = 'sklearn-fallback'

    def _tf_train_callbacks(self, mode: str, total_epochs: int) -> List[Any]:
        if tf is None:
            return []
        app = self

        class _ProgressCb(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
                logs = logs or {}
                if app._train_cancelled():
                    self.model.stop_training = True
                    return
                loss_v = logs.get('loss')
                acc_v = logs.get('accuracy', logs.get('acc'))
                parts = [f'{mode.upper()} epoch {epoch + 1}/{total_epochs}']
                if loss_v is not None:
                    parts.append(f'loss {float(loss_v):.4f}')
                if acc_v is not None:
                    parts.append(f'acc {float(acc_v):.4f}')
                app._report_train_progress(
                    mode,
                    phase='epoch',
                    epoch=epoch + 1,
                    total_epochs=total_epochs,
                    step=epoch + 1,
                    total_steps=total_epochs,
                    loss=loss_v,
                    accuracy=acc_v,
                    message=' — '.join(parts),
                )

        return [_ProgressCb()]

    def _train_tf_lstm(self, st: ModeState) -> None:
        if self._train_cancelled():
            return
        if tf is None or Sequential is None:
            self._train_sklearn_lstm(st)
            return
        X = np.asarray(st.tensor_x, dtype=np.float32)
        y = np.asarray(st.tensor_y)
        st.feature_length = int(X.shape[-1])
        labels = sorted(set(st.tensor_y))
        st.label_map = labels
        label_index = {label: i for i, label in enumerate(labels)}
        y_idx = np.asarray([label_index[v] for v in y], dtype=np.int32)
        model = Sequential()
        self.sequence_frames = int(X.shape[1])
        model.add(LSTM(32, input_shape=(self.sequence_frames, st.feature_length), return_sequences=False))
        total_epochs = 25 if len(labels) > 1 else 20
        fit_kw: Dict[str, Any] = {
            'epochs': total_epochs,
            'batch_size': min(16, len(X)),
            'shuffle': True,
            'verbose': 0,
            'callbacks': self._tf_train_callbacks('lstm', total_epochs),
        }
        self._report_train_progress(
            'lstm',
            phase='start',
            step=0,
            total_steps=total_epochs,
            message=f'LSTM training 0/{total_epochs}',
        )
        if len(labels) > 1:
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(len(labels), activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y_idx, **fit_kw)
        else:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X, np.ones((len(X), 1), dtype=np.float32), **fit_kw)
        if self._train_cancelled():
            return
        st.model = model
        st.trained = True
        st.backend = 'tensorflow'

    def _train_cancelled(self) -> bool:
        return bool(self._train_cancel)

    def _train_mode_worker(self, mode: str) -> None:
        cancelled = False
        ok = False
        try:
            st = self.mode_state(mode)
            if not st.tensor_x:
                if not self._train_cancelled():
                    self.set_notice(f'{mode.upper()} NO DATA', 'warn', sticky=True)
                return
            n = len(st.tensor_y)
            if mode in ('mlp', 'lstm') and len(set(st.tensor_y)) < 2:
                self.set_notice(
                    f'{mode.upper()} needs 2+ labels (add NONE).',
                    'warn', sticky=True,
                )
                return
            if mode == 'lstm' and len({len(x) for x in st.tensor_x}) > 1:
                self.set_notice(
                    'LSTM takes differ in length. Reset or re-record.',
                    'warn', sticky=True,
                )
                return
            if self._train_cancelled():
                cancelled = True
                return
            st.feature_length = self.get_feature_length() if mode == self.active_model else st.feature_length
            if mode == 'knn':
                self._report_train_progress(
                    'knn',
                    phase='start',
                    step=0,
                    total_steps=1,
                    message='KNN fitting…',
                )
                self._knn_fit()
                self.current_status_by_mode['knn'] = self.build_knn_status()
                self._report_train_progress(
                    'knn',
                    phase='done',
                    step=1,
                    total_steps=1,
                    message='KNN trained',
                )
            elif mode == 'mlp':
                self._train_sklearn_mlp(st)
                self.current_status_by_mode['mlp'] = self.build_mode_status('mlp')
            elif mode == 'lstm':
                self._train_tf_lstm(st)
                self.current_status_by_mode['lstm'] = self.build_mode_status('lstm')
            if self._train_cancelled():
                cancelled = True
                self.set_notice(f'{mode.upper()} training cancelled', 'warn', sticky=True)
                return
            if mode == 'knn':
                ok = self.knn_fitted
            else:
                st_done = self.mode_state(mode)
                ok = bool(st_done.trained and st_done.model)
            if ok:
                self.set_notice(
                    f'{mode.upper()} training complete — {n} sample(s). Model ready.',
                    'good',
                    sticky=True,
                )
        finally:
            cancelled = cancelled or self._train_cancel
            with self._train_lock:
                self.training = False
                self._train_cancel = False
            self.train_progress = {}
            self._emit_ws_event(
                {
                    'type': 'train_complete',
                    'ok': ok and not cancelled,
                    'message': self.last_notice,
                    'snapshot': self.snapshot(),
                }
            )

    def start_training(self, mode: str) -> str:
        if mode not in self.modes:
            return 'INVALID MODE'
        with self._train_lock:
            if self.training:
                return 'TRAINING'
            self.training = True
            self._train_cancel = False
        n = len(self.mode_state(mode).tensor_y)
        self.set_notice(f'Training {mode.upper()} on {n} sample(s)…', 'good')
        threading.Thread(target=self._train_mode_worker, args=(mode,), daemon=True).start()
        return f'Training {mode.upper()}…'

    def start_evaluation(self, mode: str) -> str:
        if mode not in self.modes:
            return 'INVALID MODE'
        if self._evaluating or self.training:
            return 'BUSY'
        self._evaluating = True
        self.set_notice(f'Evaluating {mode.upper()} (holdout)…', 'good', sticky=True)
        threading.Thread(target=self._evaluate_worker, args=(mode,), daemon=True).start()
        return f'Evaluating {mode.upper()}…'

    def _evaluate_worker(self, mode: str) -> None:
        try:
            st = self.mode_state(mode)
            if mode == 'knn':
                X, y = self._knn_companion_samples()
            else:
                X, y = list(st.tensor_x), list(st.tensor_y)
                if mode == 'lstm' and len({len(v) for v in X}) > 1:
                    raise ValueError('records have different frame lengths')
            rep_ = evaluate_holdout(mode, X, y, self.threshold) if X else {'ok': False, 'text': 'No data to evaluate.'}
            rep_['createdAt'] = now_iso()
            self.eval_reports[mode] = rep_
            self.set_notice(rep_['text'].split(chr(10))[0], 'good' if rep_['ok'] else 'warn', sticky=True)
        except Exception as exc:
            self.eval_reports[mode] = {'ok': False, 'text': f'Evaluation failed: {exc}'}
            self.set_notice(f'Evaluation failed: {exc}', 'warn', sticky=True)
        finally:
            self._evaluating = False
            self._emit_ws_event({'type': 'eval_complete', 'snapshot': self.snapshot()})

    def stop_training(self) -> None:
        self._train_cancel = True
        self.set_notice('Stopping training…', 'warn')

    def train_mode(self, mode: str) -> str:
        return self.start_training(mode)

    def add_knn_sample(self, label: str, vec: List[float]) -> None:
        st = self.mode_state('knn')
        st.feature_length = len(vec)
        st.tensor_x.append(vec)
        st.tensor_y.append(label)
        st.tensor_hand_modes.append(self.hand_mode)
        st.labels.add(label)
        st.sample_count += 1
        st.sample_buffer_count[label] = st.sample_buffer_count.get(label, 0) + 1
        self._knn_fit()
        self.current_status_by_mode['knn'] = f'RECORDED {label} #{st.sample_buffer_count[label]}'

    def add_mlp_sample(self, label: str, vec: List[float]) -> None:
        st = self.mode_state('mlp')
        st.feature_length = len(vec)
        st.tensor_x.append(vec)
        st.tensor_y.append(label)
        st.tensor_hand_modes.append(self.hand_mode)
        st.labels.add(label)
        st.sample_count += 1
        st.sample_buffer_count[label] = st.sample_buffer_count.get(label, 0) + 1
        self.current_status_by_mode['mlp'] = f'LEARNING {label} {st.sample_buffer_count[label]}/{self.target_count}'

    def add_lstm_sample(self, label: str, seq: List[List[float]]) -> None:
        st = self.mode_state('lstm')
        if seq:
            st.feature_length = len(seq[0])
        st.tensor_x.append(seq)
        st.tensor_y.append(label)
        st.tensor_hand_modes.append(self.hand_mode)
        st.labels.add(label)
        st.sample_count += 1
        st.sample_buffer_count[label] = st.sample_buffer_count.get(label, 0) + 1
        self.current_status_by_mode['lstm'] = f'RECORDED {label} #{st.sample_buffer_count[label]}'

    def _learn_negative_sequence(self, vec: List[float]) -> None:
        """NONE trong chế độ chuỗi: ghi liên tục, cửa sổ trượt chồng 50% cho tới đủ Sample target."""
        n = self.sequence_frames
        mode = 'lstm' if self.active_model == 'lstm' else 'knn'
        self.current_sequence.append(vec)
        if len(self.current_sequence) >= n:
            window = self.current_sequence[:n]
            if mode == 'lstm':
                self.add_lstm_sample(NEGATIVE_LABEL, window)
            else:
                agg = self._aggregate_sequence(window)
                if agg is not None:
                    self.add_knn_sample(NEGATIVE_LABEL, agg)
            self.current_sequence = self.current_sequence[n // 2:]
        done = self._label_count(NEGATIVE_LABEL) - self.learn_base
        self.current_sample_progress = len(self.current_sequence)
        if done >= self.target_count:
            self.learning = False
            self.current_sequence = []
            self.current_status_by_mode[mode] = f'DONE {NEGATIVE_LABEL}'
            self.set_notice(f'NONE recorded ({done} windows). Now Train / Evaluate.', 'good', sticky=True)
        else:
            self.set_notice(f'Recording NONE — move your hand freely / rest: {done}/{self.target_count} windows', 'good')

    def maybe_learn(self, vec: List[float]) -> None:
        if not self.learning or not self.current_label:
            return
        if self._uses_sequence():
            remain = self.learn_start_at - time.perf_counter()
            if remain > 0:
                self.current_sequence = []
                self.current_sample_progress = 0
                per_run = '' if self.current_label == NEGATIVE_LABEL else f' (take {self.batch_done + 1}/{self.batch_total})'
                self.set_notice(f'Get ready "{self.current_label}"{per_run} — starts in {remain:.1f}s', 'warn')
                return
        if self._uses_sequence() and self.current_label == NEGATIVE_LABEL:
            self._learn_negative_sequence(vec)
            return
        finished = False
        combo_seq = self._is_combo_knn_sequence_mode()
        combo_target = self._combo_sequence_target() if combo_seq else self.sequence_frames
        if self.active_model == 'knn':
            if combo_seq:
                self.current_sequence.append(vec)
                if len(self.current_sequence) >= combo_target:
                    sample_vec = self._aggregate_sequence(self.current_sequence[: combo_target])
                    self.current_sequence = []
                    if sample_vec is not None:
                        self.add_knn_sample(self.current_label, sample_vec)
                    self.learning = False
                    self.current_status_by_mode['knn'] = f'DONE {self.current_label}'
                    finished = True
            else:
                self.add_knn_sample(self.current_label, vec)
                st = self.mode_state('knn')
                if st.sample_buffer_count.get(self.current_label, 0) >= self.learn_base + self.target_count:
                    self.learning = False
                    self.current_status_by_mode['knn'] = f'DONE {self.current_label}'
                    finished = True
        elif self.active_model == 'mlp':
            self.add_mlp_sample(self.current_label, vec)
            st = self.mode_state('mlp')
            if st.sample_buffer_count.get(self.current_label, 0) >= self.learn_base + self.target_count:
                self.learning = False
                self.current_status_by_mode['mlp'] = f'DONE {self.current_label}'
                finished = True
        else:
            if combo_seq:
                self.current_sequence.append(vec)
                if len(self.current_sequence) >= combo_target:
                    self.add_lstm_sample(self.current_label, self.current_sequence[: combo_target])
                    self.current_sequence = []
                    self.learning = False
                    self.current_status_by_mode['lstm'] = f'DONE {self.current_label}'
                    finished = True
            else:
                self.current_sequence.append(vec)
                if len(self.current_sequence) >= self.sequence_frames:
                    self.add_lstm_sample(self.current_label, self.current_sequence[: self.sequence_frames])
                    self.current_sequence = []
                    st = self.mode_state('lstm')
                    if st.sample_buffer_count.get(self.current_label, 0) >= self.learn_base + self.target_count:
                        self.learning = False
                        self.current_status_by_mode['lstm'] = f'DONE {self.current_label}'
                        finished = True
        if self.active_model == 'lstm' or combo_seq:
            self.current_sample_progress = len(self.current_sequence)
        elif self.active_model in ('knn', 'mlp') and self.current_label:
            st2 = self.mode_state(self.active_model)
            self.current_sample_progress = st2.sample_buffer_count.get(self.current_label, 0)
        if finished and combo_seq:
            self.batch_done += 1
            if self.batch_done < self.batch_total:
                # Tự động ghi record kế tiếp sau thời gian chờ (Countdown).
                self.learning = True
                self.current_sequence = []
                self.current_sample_progress = 0
                self.learn_start_at = time.perf_counter() + self.countdown_s
                self.set_notice(
                    f'Take {self.batch_done}/{self.batch_total} saved for "{self.current_label}" — next in {self.countdown_s}s',
                    'good',
                )
                return
        if finished:
            if self.active_model == 'knn' and combo_seq:
                self.set_notice(
                    f'Sequence KNN: {self.batch_done} take(s) saved for "{self.current_label}"',
                    'good',
                    sticky=True,
                )
            elif self.active_model == 'lstm' and combo_seq:
                self.set_notice(
                    f'Sequence LSTM: {self.batch_done} take(s) saved for "{self.current_label}"',
                    'good',
                    sticky=True,
                )
            elif self.active_model == 'knn':
                self.set_notice(f'KNN capture finished for "{self.current_label}".', 'good', sticky=True)
            else:
                self.set_notice(
                    f'{self.active_model.upper()} capture finished for "{self.current_label}". '
                    f'Click Train to fit the model.',
                    'good',
                    sticky=True,
                )
        elif self.learning:
            if self.active_model == 'lstm' and not combo_seq:
                st3 = self.mode_state('lstm')
                seq_done = st3.sample_buffer_count.get(self.current_label, 0)
                self.set_notice(
                    f'Recording LSTM "{self.current_label}" — sequences {seq_done}/{self.target_count}, '
                    f'frames {len(self.current_sequence)}/{self.sequence_frames}',
                    'good',
                )
            elif combo_seq and self.active_model == 'knn':
                st4 = self.mode_state('knn')
                done = st4.sample_buffer_count.get(self.current_label, 0)
                self.set_notice(
                    f'Recording Sequence KNN "{self.current_label}" — sequences {done}, '
                    f'frames {len(self.current_sequence)}/{self.sequence_frames}',
                    'good',
                )
            elif combo_seq and self.active_model == 'lstm':
                st5 = self.mode_state('lstm')
                done = st5.sample_buffer_count.get(self.current_label, 0)
                self.set_notice(
                    f'Recording Sequence LSTM "{self.current_label}" — sequences {done}, '
                    f'frames {len(self.current_sequence)}/{self.sequence_frames}',
                    'good',
                )
            else:
                st4 = self.mode_state(self.active_model)
                done = st4.sample_buffer_count.get(self.current_label, 0)
                self.set_notice(
                    f'Recording {self.active_model.upper()} "{self.current_label}" — '
                    f'{done}/{self.target_count}',
                    'good',
                )

    def predict_vec(self, vec: List[float]) -> Tuple[str, float]:
        if self.active_model == 'knn':
            if not self._knn_model_matches_expected_length():
                return 'unknown', 0.0
            if self._is_combo_knn_sequence_mode():
                self.test_sequence.append(vec)
                if len(self.test_sequence) > self.sequence_frames:
                    self.test_sequence = self.test_sequence[-self.sequence_frames:]
                if len(self.test_sequence) < self.sequence_frames:
                    return 'unknown', 0.0
                seq_vec = self._aggregate_sequence(self.test_sequence)
                if seq_vec is None:
                    return 'unknown', 0.0
                pred = self.knn.predict(np.asarray([seq_vec], dtype=np.float32))[0]
            else:
                pred = self.knn.predict(np.asarray([vec], dtype=np.float32))[0]
            return str(pred), 1.0
        if self.active_model == 'mlp':
            st = self.mode_state('mlp')
            if not st.trained or st.model is None:
                return 'unknown', 0.0
            clf = st.model
            if hasattr(clf, 'predict_proba'):
                prob = clf.predict_proba(np.asarray([vec], dtype=np.float32))[0]
                classes = list(clf.classes_)
                idx = int(np.argmax(prob))
                return str(classes[idx]), float(prob[idx])
            pred = clf.predict(np.asarray([vec], dtype=np.float32))[0]
            return str(pred), 1.0
        st = self.mode_state('lstm')
        if not st.trained or st.model is None:
            return 'unknown', 0.0
        self.test_sequence.append(vec)
        if len(self.test_sequence) > self.sequence_frames:
            self.test_sequence = self.test_sequence[-self.sequence_frames:]
        if len(self.test_sequence) < self.sequence_frames:
            return 'unknown', 0.0
        model = st.model
        arr = np.asarray([self.test_sequence], dtype=np.float32)
        if tf is not None and st.backend == 'tensorflow' and hasattr(model, 'predict'):
            out = model.predict(arr, verbose=0)
            out = np.asarray(out)[0]
            if out.ndim == 0:
                return st.label_map[0] if st.label_map else 'unknown', float(out)
            idx = int(np.argmax(out))
            label = st.label_map[idx] if st.label_map else str(idx)
            conf = float(out[idx])
            return label, conf
        flat = arr.reshape((arr.shape[0], -1))
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(flat)[0]
            classes = list(model.classes_)
            idx = int(np.argmax(prob))
            return str(classes[idx]), float(prob[idx])
        pred = model.predict(flat)[0]
        return str(pred), 1.0

    def maybe_predict(self, vec: List[float]) -> Tuple[str, float]:
        if not self.testing:
            return 'unknown', 0.0
        return self.predict_vec(vec)
    def maybe_predict(self, vec: List[float]) -> Tuple[str, float]:
        if not self.testing:
            return 'unknown', 0.0
        return self.predict_vec(vec)

    def process_frame(self, data_url: str, client_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.update_fps()
        if isinstance(client_config, dict) and client_config:
            self.update_config(client_config)
        img = self._decode_image(data_url)
        if img is None:
            self.current_overlay = {'hands': [], 'presentSlots': {'left': False, 'right': False}}
            self.set_notice('Could not decode frame', 'warn', sticky=True)
            return {'ok': False, 'error': 'invalid_image', 'snapshot': self.snapshot()}
        if self._mp_hands is None:
            self.current_overlay = {'hands': [], 'presentSlots': {'left': False, 'right': False}}
            msg = self._hands_init_error or 'MediaPipe is not installed on the server'
            self.set_notice(msg, 'warn', sticky=True)
            return {'ok': False, 'error': 'mediapipe_missing', 'snapshot': self.snapshot()}
        detected, overlay = self._detect_hands(img)
        self.current_overlay = overlay
        if not overlay.get('hands'):
            self.stable_label = ''
            self.stable_count = 0
            self.prediction_text = 'unknown'
            self.last_prediction_label = 'unknown'
            self.last_prediction_confidence = 0.0
            if self._uses_sequence():
                self.current_sequence = []
                self.test_sequence = []
            if self.learning:
                self.set_notice('No hand detected' + (' — record restarted' if self._uses_sequence() else ''), 'warn')
            self._apply_live_status_line()
            return {'ok': True, 'overlay': overlay, 'snapshot': self.snapshot()}
        vec = self.get_frame_vector(detected)
        if vec is None:
            self.stable_label = ''
            self.stable_count = 0
            self.prediction_text = 'unknown'
            self.last_prediction_label = 'unknown'
            self.last_prediction_confidence = 0.0
            if self._uses_sequence():
                self.current_sequence = []
                self.test_sequence = []
            if self.learning:
                self.set_notice(self._hand_mode_miss_notice(), 'warn')
            self._apply_live_status_line()
            return {'ok': True, 'overlay': overlay, 'snapshot': self.snapshot()}
        if self.learning:
            self.maybe_learn(vec)
        pred_label, pred_conf = self.maybe_predict(vec)
        if self.testing:
            if pred_label != 'unknown':
                self.process_prediction(pred_label, pred_conf)
            else:
                self.stable_label = ''
                self.stable_count = 0
        self._apply_live_status_line()
        return {
            'ok': True,
            'overlay': overlay,
            'snapshot': self.snapshot(),
        }
    def start_camera(self) -> None:
        self.running = True
        self.camera_info = 'OPEN'
        self.conn_status = 'CAMERA ON'
        self.status_sticky = False
        self._apply_live_status_line()

    def stop_camera(self) -> None:
        self.running = False
        self.learning = False
        self.testing = False
        if self.training:
            self.stop_training()
        self.camera_info = 'CLOSED'
        self.conn_status = 'CAMERA OFF'
        self.current_sequence = []
        self.test_sequence = []
        self.recent_committed_labels.clear()
        self.stable_label = ''
        self.stable_count = 0
        self.current_sample_progress = 0
        self.prediction_text = '-'
        self.last_prediction_label = '-'
        self.last_prediction_confidence = 0.0
        self.status_sticky = False
        self._apply_live_status_line()

    def start_learning(self, label: str) -> None:
        self.status_sticky = False
        self.learning = True
        self.testing = False
        self.current_label = label.strip()
        self.current_sample_progress = 0
        self.learn_base = self._label_count(self.current_label)
        if self._uses_sequence():
            self.current_sequence = []
            self.learn_start_at = time.perf_counter() + self.countdown_s
            self.batch_done = 0
        self.set_notice(f'Recording {self.active_model.upper()} {self.current_label}')

    def append_learning(self, label: Optional[str] = None) -> None:
        self.status_sticky = False
        next_label = (label or self.current_label or '').strip()
        if not next_label:
            self.set_notice('Enter a gesture name first.', 'warn', sticky=True)
            return
        self.current_label = next_label
        self.learn_base = self._label_count(next_label)
        self.learning = True
        self.testing = False
        self.current_sequence = []
        self.current_sample_progress = 0
        self.learn_start_at = time.perf_counter() + self.countdown_s if self._uses_sequence() else 0.0
        self.batch_done = 0
        self.set_notice(f'Add record {self.active_model.upper()} {self.current_label}')

    def stop_learning(self) -> None:
        self.learning = False
        self.current_sequence = []
        self.set_notice('Recording stopped', 'warn')

    def start_testing(self) -> None:
        self.status_sticky = False
        self.testing = True
        self.learning = False
        self.test_sequence = []
        self.current_sample_progress = 0
        self.set_notice(f'Testing {self.active_model.upper()}')

    def stop_testing(self) -> None:
        self.testing = False
        self.test_sequence = []
        self.set_notice('Testing stopped', 'warn')

    def reset_mode(self, mode: str) -> None:
        st = self.mode_state(mode)
        st.clear()
        if mode == 'knn':
            self.knn_fitted = False
            self.knn = KNeighborsClassifier(n_neighbors=3)
        self.current_status_by_mode[mode] = f'{mode.upper()} reset'
        self.set_notice(f'{mode.upper()} reset', 'warn', sticky=True)
        if self.active_model == mode:
            self.learning = False
            self.testing = False
            self.current_label = ''
            self.current_sequence = []
            self.test_sequence = []
            self.current_sample_progress = 0
            self.prediction_text = '-'
            self.last_prediction_label = '-'
            self.last_prediction_confidence = 0.0
            self.stable_label = ''
            self.stable_count = 0
            self.recent_committed_labels = []
    def stop_learning(self) -> None:
        self.learning = False
        self.current_sequence = []
        self.set_notice('Recording stopped', 'warn')

    def start_testing(self) -> None:
        self.status_sticky = False
        self.testing = True
        self.learning = False
        self.test_sequence = []
        self.current_sample_progress = 0
        self.set_notice(f'Testing {self.active_model.upper()}')

    def stop_testing(self) -> None:
        self.testing = False
        self.test_sequence = []
        self.set_notice('Testing stopped', 'warn')

    def reset_mode(self, mode: str) -> None:
        st = self.mode_state(mode)
        st.clear()
        if mode == 'knn':
            self.knn_fitted = False
            self.knn = KNeighborsClassifier(n_neighbors=3)
        self.current_status_by_mode[mode] = f'{mode.upper()} reset'
        self.set_notice(f'{mode.upper()} reset', 'warn', sticky=True)
        if self.active_model == mode:
            self.learning = False
            self.testing = False
            self.current_label = ''
            self.current_sequence = []
            self.test_sequence = []
            self.current_sample_progress = 0
            self.prediction_text = '-'
            self.last_prediction_label = '-'
            self.last_prediction_confidence = 0.0

    def delete_label(self, mode: str, label: str) -> None:
        st = self.mode_state(mode)
        kept_x: List[Any] = []
        kept_y: List[str] = []
        kept_hm: List[str] = []
        self._ensure_tensor_hand_modes_aligned(st, self.hand_mode)
        for x, y, hm in zip(st.tensor_x, st.tensor_y, st.tensor_hand_modes):
            if y == label:
                continue
            kept_x.append(x)
            kept_y.append(y)
            kept_hm.append(hm)
        st.tensor_x = kept_x
        st.tensor_y = kept_y
        st.tensor_hand_modes = kept_hm
        st.sample_buffer_count = {}
        for y in st.tensor_y:
            k = str(y)
            st.sample_buffer_count[k] = st.sample_buffer_count.get(k, 0) + 1
        st.labels = set(st.sample_buffer_count.keys())
        st.sample_count = len(st.tensor_y)
        if mode == 'knn':
            self._knn_fit()
        elif st.model is not None:
            st.model = None
            st.trained = False
        self.current_status_by_mode[mode] = f'Deleted {label}'
        self.set_notice(f'Deleted {label} from {mode.upper()}', 'warn', sticky=True)

    def save_state(self, mode: str) -> Path:
        payload = self.serialize_mode_state(mode)
        path = STORAGE_DIR / f'{mode}-state.json'
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return path

    def load_state(self, mode: str) -> None:
        path = STORAGE_DIR / f'{mode}-state.json'
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding='utf-8'))
        self.apply_serialized_state(mode, payload)

    def serialize_mode_state(self, mode: str) -> Dict[str, Any]:
        st = self.mode_state(mode)
        self._ensure_tensor_hand_modes_aligned(st, self.hand_mode)
        model_path = None
        if mode == 'knn':
            model_path = None
        elif st.model is not None:
            model_path = f'{mode}-model.joblib' if st.backend.startswith('sklearn') else f'{mode}-model.keras'
        return {
            'type': f'gesture-{mode}-state',
            'version': 6,
            'captureProfile': self.capture_profile(),
            'createdAt': now_iso(),
            'config': {
                'threshold': self.threshold,
                'recognitionMode': self.recognition_mode,
                'sequenceFrames': self.sequence_frames,
                'targetCount': self.target_count,
            },
            'sampleCount': st.sample_count,
            'sampleBufferCount': st.sample_buffer_count,
            'sampleBufferCountByHand': self._sample_buffer_count_by_hand(st),
            'labels': sorted(list(st.labels)),
            'tensorX': st.tensor_x,
            'tensorY': st.tensor_y,
            'tensorHandModes': list(st.tensor_hand_modes),
            'trained': st.trained,
            'featureLength': st.feature_length,
            'labelMap': st.label_map,
            'backend': st.backend,
            'modelFile': model_path,
        }

    def apply_serialized_state(self, mode: str, payload: Dict[str, Any]) -> None:
        st = self.mode_state(mode)
        st.tensor_x = payload.get('tensorX', []) or []
        st.tensor_y = payload.get('tensorY', []) or []
        st.sample_count = len(st.tensor_y)
        cfg = payload.get('config')
        if not isinstance(cfg, dict):
            cfg = {}
        st.tensor_hand_modes = list(payload.get('tensorHandModes') or [])
        fb_hm = 'right'
        for h in st.tensor_hand_modes:
            if h in ('right', 'left', 'both'):
                fb_hm = str(h)
                break
        if fb_hm == 'right' and cfg.get('handMode') in ('right', 'left', 'both'):
            fb_hm = str(cfg['handMode'])
        elif fb_hm == 'right' and self.hand_mode in ('right', 'left', 'both'):
            fb_hm = self.hand_mode
        self._ensure_tensor_hand_modes_aligned(st, fb_hm)
        st.sample_buffer_count = {}
        for y in st.tensor_y:
            k = str(y)
            st.sample_buffer_count[k] = st.sample_buffer_count.get(k, 0) + 1
        st.labels = set(st.sample_buffer_count.keys())
        st.feature_length = safe_int(payload.get('featureLength', st.feature_length), st.feature_length)
        st.label_map = list(payload.get('labelMap', []) or [])
        st.trained = bool(payload.get('trained', False))
        st.backend = str(payload.get('backend', 'sklearn'))
        self.sequence_rules = []
        cap = payload.get('captureProfile')
        if isinstance(cap, dict):
            # Dùng đúng điều kiện thu của gói đã nạp để lúc test khớp lúc train.
            self.detect_width = min(1280, max(160, safe_int(cap.get('detectWidth'), self.detect_width)))
            self.max_fps = min(60, max(1, safe_int(cap.get('maxFps'), self.max_fps)))
        if isinstance(cfg, dict) and cfg:
            hm = cfg.get('handMode')
            if hm in ('right', 'left', 'both'):
                self.hand_mode = hm
            rm = cfg.get('recognitionMode')
            if rm in ('single', 'combo'):
                self.recognition_mode = rm
            self.threshold = min(1.0, max(0.0, safe_float(cfg.get('threshold'), self.threshold)))
            self.sequence_frames = max(5, safe_int(cfg.get('sequenceFrames'), self.sequence_frames))
            self.target_count = max(1, safe_int(cfg.get('targetCount'), self.target_count))
            self.stored_target_count = self.target_count
        if mode == 'knn' and st.tensor_x:
            self._knn_fit()
        elif st.trained and st.model is None and st.tensor_x:
            if mode == 'mlp':
                self._train_sklearn_mlp(st)
            else:
                self._train_tf_lstm(st)
        self.current_status_by_mode[mode] = self.build_mode_status(mode)

    @staticmethod
    def _export_basename_stamp() -> str:
        return datetime.now().strftime('%Y%m%d-%H%M%S')

    @staticmethod
    def _zip_model_bin_basename(mode: str, stamp: str, ext: str) -> str:
        return f'{stamp}-{mode}-model.{ext}'

    def _find_zip_model_bytes(
        self,
        zf: zipfile.ZipFile,
        norm: Dict[str, str],
        mode: str,
        ext: str,
    ) -> Optional[bytes]:
        pat = re.compile(rf'^\d{{8}}-\d{{6}}-{re.escape(mode)}-model\.{re.escape(ext)}$')

        def read_if_member(logical: str) -> Optional[bytes]:
            if logical not in norm:
                return None
            return zf.read(norm[logical])

        for logical in norm:
            base = logical.split('/')[-1]
            if pat.match(base):
                return zf.read(norm[logical])

        legacy = (
            (f'{mode}-model.{ext}',),
            (f'model.{ext}', f'model/model.{ext}'),
        )
        for group in legacy:
            for p in group:
                b = read_if_member(p)
                if b is not None:
                    return b
        return None

    @staticmethod
    def _zip_json_basename(mode: str, kind: str, stamp: str) -> str:
        return f'{stamp}-{mode}-{kind}.json'

    def _payload_without_feature_vectors(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(payload)
        out['tensorX'] = []
        out['tensorY'] = []
        out['tensorHandModes'] = []
        out['sampleCount'] = 0
        out['sampleBufferCount'] = {}
        return out

    @staticmethod
    def _find_zip_export_json(namelist: List[str], suffix: str) -> Optional[str]:
        pat = re.compile(
            r'^\d{8}(?:-\d{6})?-(knn|mlp|lstm)-' + re.escape(suffix) + r'$',
        )
        for raw in namelist:
            base = raw.replace('\\', '/').split('/')[-1]
            if pat.match(base):
                return raw
        return None

    def _attach_model_from_zip(
        self,
        mode: str,
        zf: zipfile.ZipFile,
        model_file_hint: Optional[str] = None,
    ) -> None:
        if mode == 'knn':
            return
        st = self.mode_state(mode)
        norm = {n.replace('\\', '/'): n for n in zf.namelist()}

        raw_keras = self._find_zip_model_bytes(zf, norm, mode, 'keras')
        raw_joblib = self._find_zip_model_bytes(zf, norm, mode, 'joblib')

        if model_file_hint:
            bn = str(model_file_hint).replace('\\', '/').split('/')[-1]
            if bn:
                for logical in norm:
                    if logical.split('/')[-1] != bn:
                        continue
                    data = zf.read(norm[logical])
                    if bn.endswith('.joblib'):
                        raw_joblib = data
                    elif bn.endswith('.keras'):
                        raw_keras = data
                    break

        def load_keras_bytes(data: bytes) -> bool:
            tmp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = Path(tmp.name)
                if tf is None:
                    return False
                st.model = tf.keras.models.load_model(str(tmp_path))
                st.trained = True
                st.backend = 'tensorflow'
                return True
            except Exception:
                return False
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

        if st.backend == 'tensorflow' and raw_keras:
            if load_keras_bytes(raw_keras):
                return

        if raw_joblib:
            try:
                st.model = joblib.load(io.BytesIO(raw_joblib))
                st.trained = True
            except Exception:
                pass
            else:
                return

        if raw_keras and load_keras_bytes(raw_keras):
            return

    def export_package(self, mode: str) -> Path:
        payload = self.serialize_mode_state(mode)
        st = self.mode_state(mode)
        stamp = self._export_basename_stamp()
        vec_name = self._zip_json_basename(mode, 'vector', stamp)
        model_name = self._zip_json_basename(mode, 'model', stamp)
        if st.model is not None and mode != 'knn':
            if st.backend.startswith('sklearn'):
                payload['modelFile'] = self._zip_model_bin_basename(mode, stamp, 'joblib')
            elif st.backend == 'tensorflow':
                payload['modelFile'] = self._zip_model_bin_basename(mode, stamp, 'keras')
        elif mode == 'knn':
            payload['modelFile'] = None
        out = EXPORT_DIR / f'gesture-{mode}-package-{stamp}.zip'
        model_cfg = self._payload_without_feature_vectors(payload)
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(vec_name, json.dumps(payload, ensure_ascii=False, indent=2))
            zf.writestr(model_name, json.dumps(model_cfg, ensure_ascii=False, indent=2))
            if st.model is not None:
                with tempfile.TemporaryDirectory() as td:
                    td_path = Path(td)
                    if mode == 'knn':
                        pass
                    elif st.backend.startswith('sklearn'):
                        model_path = td_path / 'model.joblib'
                        joblib.dump(st.model, model_path)
                        zf.write(model_path, arcname=self._zip_model_bin_basename(mode, stamp, 'joblib'))
                    elif st.backend == 'tensorflow' and tf is not None:
                        model_path = td_path / 'model.keras'
                        st.model.save(model_path)
                        if model_path.is_file():
                            zf.write(model_path, arcname=self._zip_model_bin_basename(mode, stamp, 'keras'))
            self._write_portable_files(zf, mode, stamp)
        return out

    def _write_portable_files(self, zf: zipfile.ZipFile, mode: str, stamp: str) -> None:
        """Thêm gói chạy độc lập: portable JSON, manifest, runtime NumPy, features.py."""
        st = self.mode_state(mode)
        seq = mode == 'lstm' or (mode == 'knn' and self.recognition_mode == 'combo')
        portable = build_portable(
            mode,
            backend=st.backend,
            model=st.model,
            label_map=st.label_map,
            knn_samples=self._knn_companion_samples() if mode == 'knn' else None,
            knn_k=min(3, len(st.tensor_y)) or 1,
            sequence=seq,
            sequence_frames=self.sequence_frames,
            hand_mode=self.hand_mode,
        )
        manifest = {
            'type': f'gesture-{mode}-manifest',
            'createdAt': now_iso(),
            'mode': mode,
            'handMode': self.hand_mode,
            'recognitionMode': self.recognition_mode,
            'sequenceFrames': self.sequence_frames if seq else 0,
            'threshold': self.threshold,
            'labels': sorted(st.labels) if st.labels else list(st.label_map),
            'negativeLabel': NEGATIVE_LABEL,
            'featureLength': st.feature_length,
            'captureProfile': self.capture_profile(),
            'evaluation': {k: v for k, v in (self.eval_reports.get(mode) or {}).items() if k != 'text'} or None,
            'features': (
                'Mỗi tay: 21 landmark (x,y,z) trừ cổ tay, chia cho khoảng cách cổ tay-đốt 9, '
                '+ 8 góc khớp (rad) = 71 số; hand=both ghép [left, right] = 142. Xem features.py.'
            ),
            'portableFile': f'{stamp}-{mode}-portable.json' if portable else None,
            'note': (
                'Landmark phải lấy ở đúng captureProfile (độ rộng ảnh px, nhịp khung) như lúc thu mẫu. '
                'Model chạy trên landmark đã chuẩn hoá, không trên pixel; px chỉ ảnh hưởng độ nhiễu landmark.'
            ),
        }
        zf.writestr(f'{stamp}-{mode}-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        if portable is not None:
            zf.writestr(f'{stamp}-{mode}-portable.json', json.dumps(portable, ensure_ascii=False))
            zf.write(APP_DIR / 'runtime_template.py', arcname='gesture_runtime.py')
            zf.write(APP_DIR / 'features.py', arcname='features.py')

    def _json_loads_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        text = raw_bytes.decode('utf-8-sig').strip()
        if not text:
            raise ValueError('Empty JSON file')
        return json.loads(text)

    def import_package(self, filename: str, raw_bytes: bytes) -> str:
        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(raw_bytes), 'r') as zf:
                names = zf.namelist()
                inner = self._find_zip_export_json(names, 'vector.json')
                if inner is None:
                    inner = self._find_zip_export_json(names, 'model.json')
                if inner is None and 'state.json' in names:
                    inner = 'state.json'
                if inner is None:
                    raise ValueError(
                        'ZIP must contain YYYYMMDD-[HHMMSS-]<knn|mlp|lstm>-vector.json '
                        'or YYYYMMDD-[HHMMSS-]<knn|mlp|lstm>-model.json, or legacy state.json',
                    )
                payload = self._json_loads_bytes(zf.read(inner))
                parts = str(payload.get('type') or 'gesture-knn-state').split('-')
                mode = parts[1] if len(parts) > 1 else ''
                if mode not in self.modes:
                    raise ValueError('Invalid type in ZIP JSON: %r' % (payload.get('type'),))
                self.apply_serialized_state(mode, payload)
                mf = payload.get('modelFile')
                hint = str(mf).strip() if isinstance(mf, str) and mf.strip() else None
                self._attach_model_from_zip(mode, zf, model_file_hint=hint)
                self.current_status_by_mode[mode] = self.build_mode_status(mode)
                return mode
        payload = self._json_loads_bytes(raw_bytes)
        if payload.get('type'):
            parts = str(payload['type']).split('-')
            mode = parts[1] if len(parts) > 1 else ''
            if mode not in self.modes:
                raise ValueError('Invalid type in JSON: %r' % (payload.get('type'),))
        elif payload.get('tensorX') and payload.get('tensorX') and isinstance(payload['tensorX'][0], list) and payload['tensorX'] and isinstance(payload['tensorX'][0][0], list):
            mode = 'lstm'
        else:
            mode = 'mlp'
        self.apply_serialized_state(mode, payload)
        return mode

    def handle_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        ctype = str(cmd.get('type') or '')
        if ctype == 'config':
            self.update_config(cmd)
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'start_camera':
            self.start_camera()
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'stop_camera':
            self.stop_camera()
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'start_learning':
            self.start_learning(str(cmd.get('label') or '').strip())
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'append_learning':
            self.append_learning(str(cmd.get('label') or '').strip())
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'stop_learning':
            self.stop_learning()
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'start_testing':
            self.start_testing()
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'stop_testing':
            self.stop_testing()
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'train':
            mode = str(cmd.get('mode') or self.active_model)
            msg = self.train_mode(mode)
            return {'ok': True, 'message': msg, 'snapshot': self.snapshot()}
        if ctype == 'record_negative':
            self.record_negative()
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'evaluate':
            mode = str(cmd.get('mode') or self.active_model)
            msg = self.start_evaluation(mode)
            return {'ok': True, 'message': msg, 'snapshot': self.snapshot()}
        if ctype == 'stop_training':
            self.stop_training()
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'save':
            mode = str(cmd.get('mode') or self.active_model)
            try:
                path = self.save_state(mode)
            except Exception as exc:
                err = str(exc)
                self.set_notice(f'Save failed: {err}', 'warn', sticky=True)
                return {'ok': False, 'error': err, 'snapshot': self.snapshot()}
            msg = f'{mode.upper()} saved to server: {path.name}'
            self.set_notice(msg, 'good', sticky=True)
            return {'ok': True, 'message': msg, 'snapshot': self.snapshot()}
        if ctype == 'load':
            mode = str(cmd.get('mode') or self.active_model)
            try:
                self.load_state(mode)
            except FileNotFoundError:
                err = f'No saved file for {mode.upper()} (use Save first or Import).'
                self.set_notice(err, 'warn', sticky=True)
                return {'ok': False, 'error': err, 'snapshot': self.snapshot()}
            except Exception as exc:
                err = str(exc)
                self.set_notice(f'Load failed: {err}', 'warn', sticky=True)
                return {'ok': False, 'error': err, 'snapshot': self.snapshot()}
            msg = f'{mode.upper()} loaded from storage'
            self.set_notice(msg, 'good', sticky=True)
            return {'ok': True, 'message': msg, 'snapshot': self.snapshot()}
        if ctype == 'export':
            mode = str(cmd.get('mode') or self.active_model)
            try:
                path = self.export_package(mode)
                msg = (
                    f'Exported {mode.upper()}: {path.name} — saved under backend/exports '
                    f'(no browser download; file is on the server).'
                )
                self.set_notice(msg, 'good', sticky=True)
                return {
                    'ok': True,
                    'message': msg,
                    'exportFileName': path.name,
                    'snapshot': self.snapshot(),
                }
            except Exception as exc:
                err = str(exc)
                self.set_notice(f'Export failed: {err}', 'warn', sticky=True)
                return {'ok': False, 'error': err, 'snapshot': self.snapshot()}
        if ctype == 'reset':
            mode = str(cmd.get('mode') or self.active_model)
            self.reset_mode(mode)
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'delete_label':
            mode = str(cmd.get('mode') or self.active_model)
            label = str(cmd.get('label') or '')
            self.delete_label(mode, label)
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'set_rules':
            self.update_config(cmd)
            return {'ok': True, 'snapshot': self.snapshot()}
        if ctype == 'import':
            fname = str(cmd.get('filename') or 'import.json').replace('\\', '/').split('/')[-1] or 'import.json'
            try:
                raw = base64.b64decode(str(cmd.get('contentBase64') or ''))
                if not raw:
                    raise ValueError('Empty import payload (missing or invalid base64)')
                mode = self.import_package(fname, raw)
                msg = f'Imported {mode.upper()} from {fname}'
                self.set_notice(msg, 'good', sticky=True)
                return {
                    'ok': True,
                    'message': msg,
                    'importMode': mode,
                    'importFileName': fname,
                    'snapshot': self.snapshot(),
                }
            except Exception as exc:
                err = str(exc)
                self.set_notice(f'Import failed ({fname}): {err}', 'warn', sticky=True)
                return {
                    'ok': False,
                    'error': err,
                    'importFileName': fname,
                    'snapshot': self.snapshot(),
                }
        self.set_notice(f'Unknown command: {ctype}', 'warn', sticky=True)
        return {'ok': False, 'error': f'Unknown command: {ctype}', 'snapshot': self.snapshot()}
