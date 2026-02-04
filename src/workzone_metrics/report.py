import json
import statistics
from dataclasses import asdict
from typing import Dict, Any, Optional

from .io import load_ground_truth, load_predictions
from .metrics.state import compute_state_metrics


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _stdev(values):
    values = [v for v in values if v is not None]
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def _first_state_frame(states, state: str):
    intervals = states.get(state, [])
    if not intervals:
        return None
    return min(start for start, _ in intervals)


def _add_state_start_stats(
    payload: Dict[str, Any],
    gt_states: Dict[str, Any],
    pred_states: Dict[str, Any],
    state: str,
    prefix: str,
) -> None:
    gt_start = _first_state_frame(gt_states, state)
    pred_start = _first_state_frame(pred_states, state)
    payload[f"gt_{prefix}_start_frame"] = gt_start
    payload[f"pred_{prefix}_start_frame"] = pred_start
    if gt_start is not None and pred_start is not None:
        payload[f"pred_minus_gt_{prefix}_start_frame"] = pred_start - gt_start


def generate_report(
    gt_path: str,
    pred_path: str,
    transition_tolerance_frames: int = 0,
    min_event_overlap_frames: int = 1,
) -> Dict[str, Any]:
    gt = load_ground_truth(gt_path)
    preds = load_predictions(pred_path)

    videos: Dict[str, Any] = {}
    for video, gt_entry in gt.items():
        if not gt_entry.states:
            videos[video] = {"error": "empty_ground_truth"}
            continue
        required_states = ("outside", "approaching", "inside", "exiting")
        if not all(state in gt_entry.states for state in required_states):
            videos[video] = {"error": "incomplete_ground_truth"}
            continue
        if not all(len(gt_entry.states[state]) > 0 for state in required_states):
            videos[video] = {"error": "incomplete_ground_truth"}
            continue
        pred_entry = preds.get(video)
        if pred_entry is None or pred_entry.states is None:
            videos[video] = {"error": "missing predictions or states"}
            continue
        metrics = compute_state_metrics(
            gt_entry.states,
            pred_entry.states,
            fps=pred_entry.fps,
            transition_tolerance_frames=transition_tolerance_frames,
            min_event_overlap_frames=min_event_overlap_frames,
        )
        payload = asdict(metrics)
        _add_state_start_stats(payload, gt_entry.states, pred_entry.states, "inside", "inside")
        _add_state_start_stats(payload, gt_entry.states, pred_entry.states, "approaching", "approaching")
        if pred_entry.fps is not None:
            payload["fps_estimate"] = pred_entry.fps
        videos[video] = payload

    frame_accs = [v.get("frame_accuracy") for v in videos.values() if "frame_accuracy" in v]
    trans_recalls = [v.get("transition_recall") for v in videos.values() if "transition_recall" in v]
    trans_precs = [v.get("transition_precision") for v in videos.values() if "transition_precision" in v]
    trans_accs = [v.get("transition_accuracy") for v in videos.values() if "transition_accuracy" in v]
    event_recalls = [v.get("event_recall") for v in videos.values() if "event_recall" in v]
    event_precs = [v.get("event_precision") for v in videos.values() if "event_precision" in v]
    time_err_frames = [v.get("time_in_error_frames") for v in videos.values() if "time_in_error_frames" in v]
    time_err_sec = [v.get("time_in_error_sec") for v in videos.values() if "time_in_error_sec" in v]
    entry_maes = [v.get("entry_timing_mae_frames") for v in videos.values() if "entry_timing_mae_frames" in v]
    entry_maes_sec = [v.get("entry_timing_mae_sec") for v in videos.values() if "entry_timing_mae_sec" in v]
    false_rates = [v.get("false_activation_rate") for v in videos.values() if "false_activation_rate" in v]
    mean_pers = [v.get("mean_activation_persistence_frames") for v in videos.values() if "mean_activation_persistence_frames" in v]
    mean_pers_sec = [v.get("mean_activation_persistence_sec") for v in videos.values() if "mean_activation_persistence_sec" in v]
    false_per_min = [v.get("false_activations_per_minute") for v in videos.values() if "false_activations_per_minute" in v]
    fps_estimates = [v.get("fps_estimate") for v in videos.values() if "fps_estimate" in v]
    gt_inside_starts = [v.get("gt_inside_start_frame") for v in videos.values() if "gt_inside_start_frame" in v]
    pred_inside_starts = [v.get("pred_inside_start_frame") for v in videos.values() if "pred_inside_start_frame" in v]
    pred_minus_gt = [v.get("pred_minus_gt_inside_start_frame") for v in videos.values() if "pred_minus_gt_inside_start_frame" in v]
    gt_approaching_starts = [v.get("gt_approaching_start_frame") for v in videos.values() if "gt_approaching_start_frame" in v]
    pred_approaching_starts = [v.get("pred_approaching_start_frame") for v in videos.values() if "pred_approaching_start_frame" in v]
    pred_minus_gt_approaching = [v.get("pred_minus_gt_approaching_start_frame") for v in videos.values() if "pred_minus_gt_approaching_start_frame" in v]

    summary = {
        "frame_accuracy_mean": _mean(frame_accs),
        "transition_recall_mean": _mean(trans_recalls),
        "transition_precision_mean": _mean(trans_precs),
        "transition_accuracy_mean": _mean(trans_accs),
        "event_recall_mean": _mean(event_recalls),
        "event_precision_mean": _mean(event_precs),
        "time_in_error_frames_mean": _mean(time_err_frames),
        "time_in_error_sec_mean": _mean(time_err_sec),
        "entry_timing_mae_frames_mean": _mean(entry_maes),
        "entry_timing_mae_sec_mean": _mean(entry_maes_sec),
        "false_activation_rate_mean": _mean(false_rates),
        "mean_activation_persistence_frames_mean": _mean(mean_pers),
        "mean_activation_persistence_sec_mean": _mean(mean_pers_sec),
        "false_activations_per_minute_mean": _mean(false_per_min),
        "fps_estimate_mean": _mean(fps_estimates),
        "gt_inside_start_frame_mean": _mean(gt_inside_starts),
        "gt_inside_start_frame_std": _stdev(gt_inside_starts),
        "pred_inside_start_frame_mean": _mean(pred_inside_starts),
        "pred_inside_start_frame_std": _stdev(pred_inside_starts),
        "pred_minus_gt_inside_start_frame_mean": _mean(pred_minus_gt),
        "pred_minus_gt_inside_start_frame_std": _stdev(pred_minus_gt),
        "gt_approaching_start_frame_mean": _mean(gt_approaching_starts),
        "gt_approaching_start_frame_std": _stdev(gt_approaching_starts),
        "pred_approaching_start_frame_mean": _mean(pred_approaching_starts),
        "pred_approaching_start_frame_std": _stdev(pred_approaching_starts),
        "pred_minus_gt_approaching_start_frame_mean": _mean(pred_minus_gt_approaching),
        "pred_minus_gt_approaching_start_frame_std": _stdev(pred_minus_gt_approaching),
        "videos_evaluated": len([v for v in videos.values() if "error" not in v]),
        "videos_total": len(videos),
    }

    return {"videos": videos, "summary": summary}


def write_report(report: Dict[str, Any], out_path: Optional[str]) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)
