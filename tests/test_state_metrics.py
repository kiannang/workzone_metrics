from workzone_metrics.metrics.state import compute_state_metrics


def test_state_metrics_basic():
    gt = {
        "outside": [(0, 4)],
        "approaching": [(5, 7)],
        "inside": [(8, 9)],
        "exiting": [(10, 11)],
    }
    pred = {
        "outside": [(0, 4), (12, 12)],
        "approaching": [(5, 7)],
        "inside": [(8, 9)],
        "exiting": [(10, 11)],
    }
    metrics = compute_state_metrics(gt, pred, fps=30)
    assert metrics.frame_accuracy == 1.0
    assert metrics.transition_recall == 1.0
    assert metrics.transition_precision == 1.0
    assert metrics.transition_accuracy == 1.0
    assert metrics.event_recall == 1.0
    assert metrics.event_precision == 1.0
    assert metrics.advisory_event_recall == 1.0
    assert metrics.advisory_event_precision == 1.0
    assert metrics.time_in_error_frames == 0
    assert metrics.time_in_error_sec == 0
    assert metrics.entry_timing_mae_frames == 0
    assert metrics.entry_timing_mae_sec == 0
    assert metrics.false_activation_rate == 0.0
    assert metrics.iou_inside == 1.0
    assert metrics.iou_approaching == 1.0
    assert metrics.iou_exiting == 1.0
    assert metrics.iou_outside == 1.0
    assert metrics.mean_iou == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.advisory_timing_mae_frames == 0
    assert metrics.advisory_start_error_frames == 0
    assert metrics.false_advisory_rate == 0.0
    assert metrics.simulated_speed_violation_reduction == 0.4
    assert metrics.lead_time_sec == (8 - 5) / 30
    assert metrics.late_advisory_rate == 0.0
    assert metrics.advisory_coverage_ratio == 1.0


def test_state_metrics_false_activation():
    gt = {
        "outside": [(0, 1), (5, 9)],
        "approaching": [(2, 4)],
    }
    pred = {
        "approaching": [(4, 6)],
        "outside": [(0, 3), (7, 9)],
    }
    metrics = compute_state_metrics(gt, pred, fps=30)
    assert metrics.false_activation_rate > 0
    assert metrics.false_activations_per_minute is not None
    assert metrics.false_positives_per_minute is not None
    assert metrics.iou_outside < 1.0
    assert metrics.mean_iou is not None
    assert metrics.false_advisory_rate == metrics.false_activation_rate
    assert metrics.false_advisories_per_minute == metrics.false_activations_per_minute
    assert metrics.advisory_timing_mae_frames == 2
    assert metrics.advisory_start_error_frames == 2
    assert metrics.simulated_speed_violation_reduction == (1 / 3) * 0.4
    assert metrics.lead_time_sec is None
    assert metrics.late_advisory_rate == 2 / 3
    assert metrics.advisory_coverage_ratio == 1 / 3
    assert metrics.advisory_event_recall == 1.0
    assert metrics.advisory_event_precision == 1.0


def test_state_metrics_empty_cases_are_none():
    gt = {"outside": [(0, 9)]}
    pred = {"outside": [(0, 9)]}
    metrics = compute_state_metrics(gt, pred, fps=30)
    assert metrics.transition_recall is None
    assert metrics.transition_precision is None
    assert metrics.event_recall is None
    assert metrics.event_precision is None
    assert metrics.advisory_event_recall is None
    assert metrics.advisory_event_precision is None
