"""Phụ thuộc tùy chọn: MediaPipe, TensorFlow."""

try:
    import mediapipe as mp  # type: ignore

    MEDIAPIPE_TASKS = hasattr(mp, 'tasks')
except Exception:
    mp = None
    MEDIAPIPE_TASKS = False

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import Sequential  # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, LSTM  # type: ignore
except Exception:
    tf = None
    Sequential = None
    Dense = None
    Dropout = None
    LSTM = None
