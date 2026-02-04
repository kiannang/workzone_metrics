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
    assert metrics.time_in_error_frames == 0
    assert metrics.time_in_error_sec == 0
    assert metrics.entry_timing_mae_frames == 0
    assert metrics.entry_timing_mae_sec == 0
    assert metrics.false_activation_rate == 0.0


def test_state_metrics_false_activation():
    gt = {
        "outside": [(0, 9)],
    }
    pred = {
        "approaching": [(2, 4)],
        "outside": [(0, 1), (5, 9)],
    }
    metrics = compute_state_metrics(gt, pred, fps=30)
    assert metrics.false_activation_rate > 0
    assert metrics.false_activations_per_minute is not None
    assert metrics.false_positives_per_minute is not None
