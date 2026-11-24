"""Frame-level pitch and tab metrics for TabCNN predictions.

Ported from the original Wiggins & Kim TabCNN (Python 2 + Keras) to Python 3.
The only real change from the original is replacing the Python 2 idiom
``np.array(map(fn, arr))`` (which relied on ``map`` returning a list) with an
explicit stack. TDR uses the paper's Eq. 8 count ratio rather than the
original implementation's precision ratio.

All functions take ``pred`` and ``gt`` as arrays shaped (N, 6, 21): N frames,
6 strings, 21 classes (0 = string not played, 1..20 = fret 0..19).
"""

import numpy as np

# Standard tuning, low E -> high e (MIDI note of each open string).
STRING_PITCHES = [40, 45, 50, 55, 59, 64]


def tab2pitch(tab):
    """One frame's (6, 21) tab onehots -> (44,) multi-hot pitch vector."""
    pitch_vector = np.zeros(44)
    for string_num in range(len(tab)):
        fret_class = np.argmax(tab[string_num], -1)
        # 0 means that the string is closed.
        if fret_class > 0:
            pitch_num = fret_class + STRING_PITCHES[string_num] - 41
            pitch_vector[pitch_num] = 1
    return pitch_vector


def tab2bin(tab):
    """One frame's (6, 21) tab onehots -> (6, 20) played-fret binary matrix."""
    tab_arr = np.zeros((6, 20))
    for string_num in range(len(tab)):
        fret_class = np.argmax(tab[string_num], -1)
        # 0 means that the string is closed.
        if fret_class > 0:
            fret_num = fret_class - 1
            tab_arr[string_num][fret_num] = 1
    return tab_arr


def _stack(fn, arr):
    return np.array([fn(a) for a in arr])


def pitch_precision(pred, gt):
    pitch_pred = _stack(tab2pitch, pred)
    pitch_gt = _stack(tab2pitch, gt)
    numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
    denominator = np.sum(pitch_pred.flatten())
    return (1.0 * numerator) / denominator


def pitch_recall(pred, gt):
    pitch_pred = _stack(tab2pitch, pred)
    pitch_gt = _stack(tab2pitch, gt)
    numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
    denominator = np.sum(pitch_gt.flatten())
    return (1.0 * numerator) / denominator


def pitch_f_measure(pred, gt):
    p = pitch_precision(pred, gt)
    r = pitch_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f


def tab_precision(pred, gt):
    # Drop the "closed" class, as we only want to count positives.
    tab_pred = _stack(tab2bin, pred)
    tab_gt = _stack(tab2bin, gt)
    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_pred.flatten())
    return (1.0 * numerator) / denominator


def tab_recall(pred, gt):
    # Drop the "closed" class, as we only want to count positives.
    tab_pred = _stack(tab2bin, pred)
    tab_gt = _stack(tab2bin, gt)
    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_gt.flatten())
    return (1.0 * numerator) / denominator


def tab_f_measure(pred, gt):
    p = tab_precision(pred, gt)
    r = tab_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f


def tab_disamb(pred, gt):
    """Paper Eq. 8: correct string-fret detections / correct distinct pitches."""
    tab_tp = np.sum(_stack(tab2bin, pred) * _stack(tab2bin, gt))
    pitch_tp = np.sum(_stack(tab2pitch, pred) * _stack(tab2pitch, gt))
    return float(tab_tp / pitch_tp) if pitch_tp else 0.0
