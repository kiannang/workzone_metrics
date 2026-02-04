# Workzone Metrics Harness

## Latest Report Summary
**Filter:** Only videos with all four GT states present and non-empty (190 / 692).

**report.json (tolerance = 0 frames)**
- Frame accuracy: **0.4771**
- Entry timing MAE: **293.07 frames (9.18s)**
- Event precision / recall (INSIDE): **0.7421 / 0.9711**
- Transition precision / recall: **0.00194 / 0.00351**
- False activation rate (frame-level): **0.5694**
- False activations / minute: **5.17**
- Time-in-error: **501.66 frames (15.70s)**

**report_tolerance5.json (tolerance = 5 frames)**
- Frame accuracy: **0.4771**
- Entry timing MAE: **293.07 frames (9.18s)**
- Event precision / recall (INSIDE): **0.7421 / 0.9711**
- Transition precision / recall: **0.0182 / 0.0353**
- False activation rate (frame-level): **0.5694**
- False activations / minute: **5.17**
- Time-in-error: **501.66 frames (15.70s)**

This repo provides a minimal evaluation harness for state-based workzone metrics. It is wired to your ground-truth JSON format and expects a matching predictions JSON format.

## Ground-truth JSON (current)
Your uploaded `workzone_annotations.json` is a dictionary keyed by video snippet name, with state intervals in **inclusive** frame indices:

```json
{
  "video_snippet.mp4": {
    "outside": [[0, 150], [423, 900]],
    "approaching": [[151, 228]],
    "inside": [[229, 338]],
    "exiting": [[339, 422]]
  }
}
```

## Predictions inputs
You can provide either:
- A predictions JSON with interval states, or
- A workzone timeline CSV (or a directory of `*_timeline*.csv` files) from `process_video_fusion.py`.

### Predictions JSON (intervals)
Create a predictions JSON with the same keys and a `states` object in the same interval format. Optional fields `fps`, `detections`, and `ocr` are reserved for future metrics.

```json
{
  "video_snippet.mp4": {
    "fps": 30,
    "states": {
      "outside": [[0, 149], [430, 900]],
      "approaching": [[150, 230]],
      "inside": [[231, 340]],
      "exiting": [[341, 429]]
    },
    "detections": null,
    "ocr": null
  }
}
```

## Metrics implemented (state-based)
- Frame accuracy (state per frame)
- State transition precision/recall (configurable tolerance in frames)
- Entry timing MAE (frames)
- False work-zone activation rate (fraction of GT-outside frames predicted non-outside)
- Mean activation persistence (frames)
- False activations per minute (requires `fps`)

## Metric Definitions (How We Calculate)
This section documents the exact calculation logic used by the harness. Inputs are:
- **GT**: `workzone_annotations.json` with inclusive frame intervals per state.
- **Predictions**: timeline CSV(s) with per-frame `frame` and `state` (and optionally `time_sec`).

### Frame Accuracy
**Definition**: Percentage of frames where predicted state equals GT state.

**Calculation**:
- Convert GT intervals to a per-frame label array.
- Convert predicted intervals to a per-frame label array.
- `frame_accuracy = correct_frames / total_frames`.

### State Transition Precision / Recall / Accuracy
**Definition**: How well predicted state changes line up with GT transitions.

**Transitions**:
- A transition is a change in label between consecutive frames: `(from_state → to_state)` at frame `t`.

**Matching rule**:
- A predicted transition matches a GT transition if:
  - `(from_state, to_state)` are identical, and
  - `abs(pred_frame - gt_frame) <= tolerance_frames`.

**Metrics**:
- `transition_recall = matched / gt_transitions`
- `transition_precision = matched / predicted_transitions`
- `transition_accuracy = matched / max(gt_transitions, predicted_transitions)`

### Entry Timing MAE
**Definition**: Error in the first INSIDE frame.

**Calculation**:
- `gt_entry = first GT frame labeled INSIDE`
- `pred_entry = first predicted frame labeled INSIDE`
- `entry_timing_mae_frames = abs(pred_entry - gt_entry)`
- `entry_timing_mae_sec = entry_timing_mae_frames / fps` (if `time_sec` is present)

### Frame-level False Activation Rate
**Definition**: Fraction of GT OUTSIDE frames predicted as not-OUTSIDE.

**Calculation**:
- `false_activation_rate = (# frames where GT=outside and Pred≠outside) / (# GT outside frames)`

### Mean Activation Persistence
**Definition**: Average length of continuous predicted non-OUTSIDE runs.

**Calculation**:
- Find contiguous segments where Pred≠outside.
- `mean_activation_persistence_frames = mean(segment_lengths)`
- `mean_activation_persistence_sec = mean_activation_persistence_frames / fps` (if available)

### False Activations per Minute
**Definition**: Rate of false activation events per minute.

**Calculation**:
- A false activation event is a transition into Pred≠outside while GT=outside.
- `false_activations_per_minute = false_events / total_minutes`

### Work-Zone Event Precision / Recall (INSIDE)
**Definition**: Measures whether a work-zone is detected at all, independent of exact timing.

**Event definition**:
- A GT event is a contiguous GT `inside` interval.
- A predicted event is a contiguous predicted `inside` interval.

**Matching rule**:
- A predicted event matches a GT event if their temporal overlap ≥ `min_overlap_frames`.

**Metrics**:
- `event_recall = matched_gt_events / total_gt_events`
- `event_precision = matched_pred_events / total_pred_events`

CLI flag:
- `--min-event-overlap-frames` (default `1`)

**Recommendation**:
- For our use case, **event-level precision/recall is the primary metric** to judge if a work-zone is detected at all.

### Time-in-Error
**Definition**: Total duration where predicted state ≠ GT state.

**Calculation**:
- `time_in_error_frames = total_frames - correct_frames`
- `time_in_error_sec = time_in_error_frames / fps`

### FPS Estimate
**Definition**: Estimated frames-per-second from timeline timestamps.

**Calculation**:
- For each adjacent pair of rows: `fps_i = delta_frames / delta_time_sec`.
- `fps_estimate = median(fps_i)`.

## Metrics pending data/schema
- mAP@0.5 (detection)
- Precision @ high recall (detection)
- OCR sign accuracy
- False positives / minute (detection-driven)
- FPS (runtime measurement vs. reported)

### Timeline CSV input
The workzone timeline CSV includes per-frame `state`, `frame`, and `time_sec`. The CLI will parse those into state intervals and estimate FPS from `time_sec`.

## Run
```bash
python -m workzone_metrics.cli --gt workzone_annotations.json --pred predictions.json --out results/report.json
# or directly from timeline CSV(s)
python -m workzone_metrics.cli --gt workzone_annotations.json --pred workzone-main/workzone-main/outputs --out results/report.json
```

## COCO Detection Eval (mAP@0.5)
This requires `torch`, `ultralytics`, and `pycocotools`. In this environment, package downloads are blocked, so install these locally or provide wheels.

Example (validation split):
```bash
python scripts/coco_eval_yolo.py \\
  --gt ROADWORK_data/annotations/annotations/instances_val_gps_split_with_signs.json \\
  --images ROADWORK_data/images/images \\
  --weights workzone-main/workzone-main/weights/yolo12s_hardneg_1280.pt \\
  --out results/coco_preds_val.json \\
  --summary results/coco_metrics_val.json \\
  --imgsz 1280 --conf 0.25 --iou 0.45 --device cpu
```

## Notes
- Frame intervals are treated as inclusive `[start, end]`.
- If a video is missing predictions, it is reported with an error entry.
- If a GT entry has no intervals for any state, it is skipped with `empty_ground_truth`.
- If a GT entry does not include all four states (`outside`, `approaching`, `inside`, `exiting`) with non-empty intervals, it is skipped with `incomplete_ground_truth`.
 
## Why Some Videos Are Skipped
We skip certain videos to avoid misleading metrics:
- **`empty_ground_truth`**: The GT has no intervals at all, so there is nothing to score.
- **`incomplete_ground_truth`**: We require all four states to be present with at least one interval. Missing states make transition/timing metrics ambiguous and inflate accuracy.
- **`missing predictions or states`**: The model did not produce a timeline CSV or state data for that video, so metrics can’t be computed.
