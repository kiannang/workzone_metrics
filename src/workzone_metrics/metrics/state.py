from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

StateIntervals = Dict[str, List[Tuple[int, int]]]

DEFAULT_STATE_ORDER = ["inside", "exiting", "approaching", "outside"]


@dataclass
class StateMetrics:
    frame_accuracy: float
    transition_recall: float
    transition_precision: float
    transition_accuracy: float
    event_recall: float
    event_precision: float
    time_in_error_frames: int
    time_in_error_sec: Optional[float]
    entry_timing_mae_frames: Optional[float]
    entry_timing_mae_sec: Optional[float]
    false_activation_rate: float
    mean_activation_persistence_frames: float
    mean_activation_persistence_sec: Optional[float]
    false_activations_per_minute: Optional[float]
    false_positives_per_minute: Optional[float]


def _max_frame(states: StateIntervals) -> int:
    max_end = 0
    for intervals in states.values():
        for _, end in intervals:
            if end > max_end:
                max_end = end
    return max_end


def _labels_from_intervals(
    states: StateIntervals,
    total_frames: int,
    order: List[str] = None,
    default_label: str = "outside",
) -> List[str]:
    if order is None:
        order = DEFAULT_STATE_ORDER
    labels = [default_label] * total_frames
    for state in order:
        intervals = states.get(state, [])
        for start, end in intervals:
            start = max(0, start)
            end = min(total_frames - 1, end)
            for i in range(start, end + 1):
                labels[i] = state
    return labels


def _transitions(labels: List[str]) -> List[Tuple[str, str, int]]:
    transitions: List[Tuple[str, str, int]] = []
    if not labels:
        return transitions
    prev = labels[0]
    for idx in range(1, len(labels)):
        cur = labels[idx]
        if cur != prev:
            transitions.append((prev, cur, idx))
            prev = cur
    return transitions


def _match_transitions(
    gt: List[Tuple[str, str, int]],
    pred: List[Tuple[str, str, int]],
    tolerance: int,
) -> Tuple[int, int, int]:
    matched = 0
    used = [False] * len(pred)
    for g_from, g_to, g_frame in gt:
        for i, (p_from, p_to, p_frame) in enumerate(pred):
            if used[i]:
                continue
            if p_from == g_from and p_to == g_to and abs(p_frame - g_frame) <= tolerance:
                used[i] = True
                matched += 1
                break
    return matched, len(gt), len(pred)


def _first_state_frame(states: StateIntervals, state: str) -> Optional[int]:
    intervals = states.get(state, [])
    if not intervals:
        return None
    return min(start for start, _ in intervals)


def compute_state_metrics(
    gt_states: StateIntervals,
    pred_states: StateIntervals,
    fps: Optional[float] = None,
    transition_tolerance_frames: int = 0,
    entry_state: str = "inside",
    outside_state: str = "outside",
    min_event_overlap_frames: int = 1,
) -> StateMetrics:
    total_frames = max(_max_frame(gt_states), _max_frame(pred_states)) + 1
    if total_frames <= 0:
        total_frames = 1

    gt_labels = _labels_from_intervals(gt_states, total_frames)
    pred_labels = _labels_from_intervals(pred_states, total_frames)

    correct = sum(1 for g, p in zip(gt_labels, pred_labels) if g == p)
    frame_accuracy = correct / total_frames
    time_in_error_frames = total_frames - correct

    gt_trans = _transitions(gt_labels)
    pred_trans = _transitions(pred_labels)
    matched, gt_count, pred_count = _match_transitions(gt_trans, pred_trans, transition_tolerance_frames)
    transition_recall = matched / gt_count if gt_count else 1.0
    transition_precision = matched / pred_count if pred_count else 1.0
    denom = max(gt_count, pred_count)
    transition_accuracy = matched / denom if denom else 1.0

    gt_events = gt_states.get(entry_state, [])
    pred_events = pred_states.get(entry_state, [])
    matched_events = _match_events(gt_events, pred_events, min_event_overlap_frames)
    event_recall = matched_events / len(gt_events) if gt_events else 1.0
    event_precision = matched_events / len(pred_events) if pred_events else 1.0

    gt_entry = _first_state_frame(gt_states, entry_state)
    pred_entry = _first_state_frame(pred_states, entry_state)
    entry_timing_mae = None
    if gt_entry is not None and pred_entry is not None:
        entry_timing_mae = abs(pred_entry - gt_entry)

    false_activation_frames = 0
    gt_outside_frames = 0
    for g, p in zip(gt_labels, pred_labels):
        if g == outside_state:
            gt_outside_frames += 1
            if p != outside_state:
                false_activation_frames += 1
    false_activation_rate = (
        false_activation_frames / gt_outside_frames if gt_outside_frames else 0.0
    )

    activation_lengths: List[int] = []
    run = 0
    for p in pred_labels:
        if p != outside_state:
            run += 1
        elif run:
            activation_lengths.append(run)
            run = 0
    if run:
        activation_lengths.append(run)
    mean_persistence = sum(activation_lengths) / len(activation_lengths) if activation_lengths else 0.0

    false_activations_per_minute = None
    entry_timing_mae_sec = None
    mean_persistence_sec = None
    time_in_error_sec = None
    if fps and fps > 0:
        total_minutes = total_frames / fps / 60.0
        false_activation_events = 0
        in_false = False
        for g, p in zip(gt_labels, pred_labels):
            is_false = g == outside_state and p != outside_state
            if is_false and not in_false:
                false_activation_events += 1
            in_false = is_false
        false_activations_per_minute = false_activation_events / total_minutes if total_minutes else 0.0
        if entry_timing_mae is not None:
            entry_timing_mae_sec = entry_timing_mae / fps
        mean_persistence_sec = mean_persistence / fps
        time_in_error_sec = time_in_error_frames / fps

    return StateMetrics(
        frame_accuracy=frame_accuracy,
        transition_recall=transition_recall,
        transition_precision=transition_precision,
        transition_accuracy=transition_accuracy,
        event_recall=event_recall,
        event_precision=event_precision,
        time_in_error_frames=time_in_error_frames,
        time_in_error_sec=time_in_error_sec,
        entry_timing_mae_frames=entry_timing_mae,
        entry_timing_mae_sec=entry_timing_mae_sec,
        false_activation_rate=false_activation_rate,
        mean_activation_persistence_frames=mean_persistence,
        mean_activation_persistence_sec=mean_persistence_sec,
        false_activations_per_minute=false_activations_per_minute,
        false_positives_per_minute=false_activations_per_minute,
    )


def _overlap_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0, end - start + 1)


def _match_events(
    gt_events: List[Tuple[int, int]],
    pred_events: List[Tuple[int, int]],
    min_overlap_frames: int,
) -> int:
    matched = 0
    used = [False] * len(pred_events)
    for g in gt_events:
        for i, p in enumerate(pred_events):
            if used[i]:
                continue
            if _overlap_len(g, p) >= min_overlap_frames:
                used[i] = True
                matched += 1
                break
    return matched
