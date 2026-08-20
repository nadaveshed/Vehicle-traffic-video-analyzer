# Vehicle Closest-Approach Detector

A Python system that receives a road-traffic video and reports, for every tracked vehicle:

- The frame in which the vehicle was closest to the camera.
- The timestamp of that moment.
- The vehicle position as a bounding box.
- An estimated distance from the camera.
- Numeric and visual evidence that helps validate the selected frame.

The project was developed as a solution to a 90-minute Detection & Tracking assignment. The main sample is `data/highway.mp4`, and `data/highway_dense_4k.mp4` is used as a more difficult stress test.

---

## Running the Project

### Requirements

- Python 3.11 or newer.
- An NVIDIA GPU is recommended, but CPU execution is also supported.
- An input format supported by OpenCV, such as MP4.

### Windows installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Select a Torch build that matches the installed CUDA version.
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
pip install -r requirements.txt
```

### Process the main sample

```powershell
.\.venv\Scripts\python.exe main.py process data\highway.mp4 --open
```

### Process the dense 4K sample

```powershell
.\.venv\Scripts\python.exe main.py process data\highway_dense_4k.mp4
```

### Print the results of an existing job

```powershell
.\.venv\Scripts\python.exe main.py results 2
```

### Start the REST API

```powershell
.\.venv\Scripts\python.exe main.py serve
```

Swagger documentation is then available at:

```text
http://127.0.0.1:8000/docs
```

Each execution creates a new `output/job_XXXX` directory.

---

## Result at a Glance

The system reads the video frame by frame, detects vehicles with YOLO11s, and uses ByteTrack to assign a persistent identifier to each vehicle. It then builds a position-and-size history for every track and selects the frame with the lowest estimated camera distance.

```text
Video
  -> OpenCV
  -> YOLO11s detection
  -> ByteTrack tracking
  -> detections grouped by track_id
  -> camera calibration
  -> closest-frame calculation
  -> CSV / JSON / plots / HTML / annotated video
```

A simplified result for one vehicle looks like this:

```json
{
  "track_id": 24,
  "class": "car",
  "closest_frame": 326,
  "timestamp_hms": "00:10.869",
  "x1": 892,
  "y1": 1524,
  "x2": 1662,
  "y2": 2135,
  "distance_m": 9.9
}
```

---

## Main Components in Plain Language

### What is YOLO11s?

YOLO11s is a ready-to-use object-detection model. It can be treated as a function that receives an image and returns a list such as:

```text
car, confidence=0.91, box=[x1, y1, x2, y2]
truck, confidence=0.87, box=[x1, y1, x2, y2]
```

The `s` means Small. It offers a practical balance between speed and accuracy: it is more capable than the `n` variant while being faster and lighter than the `m`, `l`, and `x` variants.

The project loads the local `yolo11s.pt` weights. The model was pre-trained on COCO and already recognizes `car`, `motorcycle`, `bus`, and `truck`; no additional training was performed.

YOLO does not know that a vehicle in the current frame is the same vehicle seen in the previous frame. It only detects objects in each image independently.

### What is ByteTrack?

ByteTrack connects YOLO detections over time. If the same car appears in 80 frames, ByteTrack should assign the same `track_id` in all 80 frames.

```text
frame 100: car, track_id=17
frame 101: car, track_id=17
frame 102: car, track_id=17
```

This creates a per-vehicle array of positions that can be analyzed across the video.

---

## How Is the Closest Frame Selected?

A normal video contains pixels but no depth sensor. Therefore, exact physical distance cannot be recovered from a single uncalibrated camera. The system uses visual perspective cues instead:

- A nearby vehicle appears wider and larger.
- Its bottom edge moves lower in the image.
- Its box occupies a larger image area.

For every detection, the estimated road-contact point is the bottom center of its bounding box:

```text
contact_x = (x1 + x2) / 2
contact_y = y2
```

This point is preferred over the box center because it approximately represents where the vehicle meets the road.

### Why not select the largest bounding box?

That is the simplest approach, but it fails as a vehicle leaves through the bottom of the frame. At that point, only part of the vehicle remains visible, so YOLO returns a box for the visible fragment. The box area may shrink even though the vehicle is still approaching.

The system therefore calculates three distance estimates:

1. `ground` — proximity based on the road-contact position.
2. `width` — proximity based on the vehicle width in pixels.
3. `diag` — proximity based on box width, height, and area.

In code terms, each estimator produces an array with one value per frame in the track. The system takes the median of the available estimates, smooths small detection jumps, and selects the minimum:

```text
closest_frame = argmin(smoothed_distance)
timestamp = closest_frame / fps
```

When a bounding box touches an image border, that sample is marked as cropped and excluded from the distance calculation because the full vehicle dimensions are unknown.

The exact definition used by the system is therefore:

> The closest frame in which the vehicle is still fully visible and can be measured reliably.

---

## Camera Calibration

No camera height, focal length, or depth map was provided. Instead of requiring manual calibration, the system learns the scene perspective from the video detections.

It compares:

- The vehicle width in pixels.
- The vertical position of its road-contact point.

Detections are grouped into width bins, and the median contact position is calculated for each bin. A monotonic curve is then constructed: a wider vehicle cannot be considered farther away.

Calibration is mainly used to determine the shape of the distance curve. Absolute distance in meters remains an estimate because it also depends on:

- An assumed horizontal field of view of 50 degrees.
- Typical widths and heights for each vehicle class.

For this reason, the selected frame and timestamp are more trustworthy than the absolute distance in meters.

---

## Capture Zone

The system places a virtual rectangle over the part of the road nearest to the camera. A vehicle becomes `armed` when its road-contact point enters the rectangle.

The default values are defined in `config/settings.py`:

```text
x=0.15, y=0.55, width=0.85, height=0.45
```

The coordinates are normalized between 0 and 1 rather than stored as pixels, so the same zone definition works across different video resolutions.

The zone can be changed from the command line:

```powershell
.\.venv\Scripts\python.exe main.py process data\highway.mp4 `
  --zone 0.15 0.55 0.85 0.45
```

The zone reduces irrelevant selections from distant vehicles or unrelated lanes. A track that never enters the zone can still be analyzed, but the result is marked with `entered_zone=false` so it can be filtered later.

---

## Architecture and Data Flow

The project is split into layers so that each component has one responsibility:

```text
CLI / REST API
      |
      v
PipelineService
      |
      +--> TrackingService ------> YOLO + ByteTrack
      +--> CalibrationService ---> camera model
      +--> ZoneService ----------> capture window
      +--> ProximityService -----> closest frame
      +--> VisualizationService -> annotated media
      +--> PlotService ----------> numeric plots
      +--> ReportService --------> HTML report
      |
      v
Repositories + SQLite + output files
```

This architecture was selected because:

- YOLO or ByteTrack can be replaced without rewriting the proximity logic.
- The API contains no business logic.
- Geometry calculations do not depend on FastAPI, SQLite, or OpenCV I/O.
- Every stage can be tested separately.
- The same pipeline is used by both the command-line interface and REST API.
- Tunable parameters live in one configuration file instead of being scattered through the code.

---

## Important Files

### Entry point and configuration

| File | Responsibility |
|---|---|
| `main.py` | Implements the `process`, `results`, `reanalyse`, and `serve` commands. |
| `requirements.txt` | Lists the required Python packages. |
| `config/settings.py` | Contains the model, thresholds, capture zone, vehicle dimensions, and output paths. |
| `yolo11s.pt` | Pre-trained object-detection weights. |

### Core

| File | Responsibility |
|---|---|
| `core/models.py` | Defines detections, video metadata, calibration data, and closest-approach results. |
| `core/geometry.py` | Implements bounding-box calculations, road-contact points, crop detection, and interpolation. |

### Services

| File | Responsibility |
|---|---|
| `services/video_service.py` | Reads the video, FPS, dimensions, and frames. |
| `services/tracking_service.py` | Runs YOLO11s and ByteTrack and converts their output to internal records. |
| `services/zone_service.py` | Determines when a vehicle enters and exits the capture zone. |
| `services/calibration_service.py` | Builds the camera-perspective model from video detections. |
| `services/proximity_service.py` | Calculates the three estimates and selects the closest frame. |
| `services/pipeline_service.py` | Orchestrates the stages and passes data between them. |
| `services/visualization_service.py` | Creates the annotated video, snapshots, crops, and contact sheets. |
| `services/plot_service.py` | Creates per-vehicle and validation plots. |
| `services/report_service.py` | Builds the HTML report. |
| `services/detection_cache.py` | Stores detection results for re-analysis without rerunning YOLO. |

### Persistence and API

| Directory | Responsibility |
|---|---|
| `repository/` | SQLite access and persistence of jobs and vehicle events. |
| `api/` | FastAPI routes and schemas. |
| `tests/` | Automated tests for the central logic. |
| `data/` | Input videos. |
| `output/` | Generated run artifacts. |

---

## Output Files

A full execution may create:

| Output | Contents |
|---|---|
| `results.csv` | One row per track, suitable for Excel. |
| `results.json` | The same results together with calibration and validation data. |
| `annotated.mp4` | Video with boxes, track IDs, the capture zone, and closest-frame markers. |
| `report.html` | A central report that can be opened in a browser. |
| `plots/track_NNN.png` | The proximity curve for one vehicle. |
| `plots/calibration_fit.png` | A visualization of the camera calibration. |
| `plots/validation_agreement.png` | Agreement between estimators and comparison with the simple baseline. |
| `snapshots/` | The selected frame for every vehicle. |
| `crops/` | The vehicle cropped from its selected frame. |
| `contact_sheets/` | Frames before and after the selected moment for visual inspection. |

The result table includes fields such as:

- `track_id`
- `class`
- `closest_frame`
- `timestamp_hms`
- `x1, y1, x2, y2`
- `distance_m`
- `confidence`
- `entered_zone`
- `agreement_frames`
- `naive_area_frame`

---

## System Accuracy and Measurement

Time resolution, internal consistency, and true accuracy against ground truth are different concepts.

### Time resolution

`highway_dense_4k.mp4` contains approximately 30 frames per second. Therefore, the time interval between frames is:

```text
1 / 30 = 0.0333 seconds
```

The selected frame therefore has a time resolution of approximately 33 milliseconds. The system also calculates a refined timestamp through interpolation, but it cannot create visual information that does not exist between source frames.

### Dense 4K run results

| Metric | Result |
|---|---:|
| Reported tracks | 202 |
| Tracks that entered the capture zone | 65 |
| Calibration samples | 25,946 |
| Calibration fit, R² | 0.9506 |
| Median spread between the three estimates | 11 frames |
| Median spread in time | 0.367 seconds |
| Estimates within one frame | 27.7% |
| Estimates within three frames | 35.1% |
| Disagreement with largest-box baseline | 70.3% |

`R²=0.9506` means that the calibration model fits its samples well. It does **not** mean that the entire system is 95% accurate.

`agreement_frames` measures the distance between the frames independently selected by the three proximity estimators. A small value increases confidence in the selection. However, all three estimates originate from the same detections, so this is internal consistency rather than external ground-truth accuracy.

### What is required for a true accuracy percentage?

No ground-truth file with manually selected closest frames was provided. It would therefore be misleading to claim an absolute accuracy percentage.

A correct external evaluation would manually label a representative set of vehicles and calculate:

- Mean absolute error in frames.
- Median absolute error in frames.
- Percentage of results within ±1 and ±3 frames.
- Number of track-ID switches and track fragments.
- Agreement between two human annotators.

---

## Critical Analysis

### Where the system performs well

- Fixed cameras with vehicles approaching in a clear direction.
- Multiple vehicles can be processed simultaneously.
- No custom model training is required.
- Selection does not depend only on bounding-box area.
- Cropped detections at image boundaries are explicitly identified.
- Camera calibration is derived from the video rather than requiring a site survey.
- All required values are produced: frame, timestamp, and position.
- CSV/JSON results and visual evidence are both available.
- Components are separated and can be replaced or tested independently.

### Where the system is weaker

- In dense traffic, ByteTrack may lose a vehicle and assign it a new ID. One physical vehicle may then appear more than once in the results.
- Occlusion by another vehicle harms both detection and tracking.
- A track that begins at the first frame or ends at the final frame is incomplete; its true closest moment may be outside the clip.
- Only 65 of the 202 tracks in the dense 4K run entered the capture zone. Results with `entered_zone=true` should therefore be presented separately.
- Estimator agreement was relatively weak in the dense clip: only 35.1% were within three frames.
- Distance in meters depends on assumed field of view and typical vehicle dimensions, so it is less reliable than the selected frame.
- A typical vehicle width cannot accurately describe every car, bus, or truck.
- A road with changing slopes or multiple elevations does not perfectly fit a single-road-surface model.
- One dense-run track produced a negative distance. This is physically invalid and indicates extrapolation outside the calibration range; it should be rejected.
- Job 2 produced the annotated video, CSV, JSON, and plots, but its snapshot and contact-sheet directories were not retained. This is an artifact-generation issue rather than a detection failure.

---

## How the Detection Could Be Taken to a Higher Level

The current pipeline is a single-pass, off-the-shelf configuration: stock YOLO11s weights, stock ByteTrack, one geometric depth model. The improvements below are ordered by expected gain per unit of effort, and each one names the concrete weakness it removes.

### 1. Detector quality

| Change | Why it helps | Cost |
|---|---|---|
| Move to `yolo11m` / `yolo11l`, or run `yolo11s` at `imgsz=1280` on the 4K clip | Small vehicles far from the camera are the main source of late track starts. A track that only begins after the vehicle is already close has fewer samples for the calibration curve and a shorter approach profile. Higher input resolution recovers those early detections without any training. | GPU time only; `imgsz` is already a setting. |
| Tile the frame (SAHI-style sliced inference) on 4K input | Instead of downscaling 3840 px to 960 px, run inference on overlapping crops and merge. Distant vehicles keep their pixel detail; the downscale is what destroys them today. | ~3–4× slower per frame; worth it offline, not for live use. |
| Test-time augmentation (horizontal flip + multi-scale) | Raises recall a few points on partly occluded boxes, which is exactly where tracks break. | ~2–3× slower; use only on the evaluation set. |
| Fine-tune on a few hundred labelled frames from the target camera | COCO cars are photographed from the side and from eye level. This camera looks down a highway. Fine-tuning on in-domain frames mostly fixes box *tightness*, and box tightness is what every depth estimate here is computed from — a 5% width error is a 5% distance error. | The largest item on this list: labelling, a training run, and a held-out split. |
| Calibrate `conf_threshold` per class instead of one global `0.35` | Motorcycles and distant trucks score lower than cars. One threshold either drops them or admits car false positives. | Minutes, once a labelled set exists. |

### 2. Tracking robustness

This is where the dense 4K run actually loses accuracy: 202 reported tracks for a scene that does not contain 202 distinct vehicles.

- **Switch to BoT-SORT with ReID.** ByteTrack associates on motion and IoU alone. When a car is occluded by a truck for 10 frames, the IoU chain breaks and a new `track_id` is issued. BoT-SORT adds an appearance embedding, so the vehicle is re-attached to its original id after the occlusion. This is a one-line change to `tracker_config` plus a ReID weight file, and it directly attacks the "one physical vehicle appears more than once" failure listed above.
- **Camera-motion compensation.** Cheap insurance if the camera is ever handheld or mounted on a pole that vibrates; BoT-SORT's GMC handles it.
- **Track stitching as a post-process.** After tracking, merge two tracks when the tail of A and the head of B are close in time, spatially consistent with A's velocity, and share a class. This is pure geometry on data that already exists, and it can be evaluated offline against the same detections.
- **Raise `min_track_length` and reject boundary tracks.** A track that starts at frame 0 or ends at the last frame has its true closest approach outside the clip. Reporting it at all is a guess; it should be flagged and excluded from headline metrics.
- **Kalman state on the road-contact point, not the box centre.** The contact point moves smoothly along the road under perspective; the box centre does not, because it drifts upward as the vehicle is cropped at the bottom edge.

### 3. Depth estimation

The three estimators (`ground`, `width`, `diag`) all consume the same bounding box, so they share its failure modes — which is why their agreement is internal consistency, not accuracy.

- **Add a genuinely independent estimator.** A monocular depth network (Depth Anything V2, Metric3D) sampled at the road-contact point gives a depth signal that does not come from the box dimensions at all. Adding it turns the median fusion from "three views of one measurement" into a real vote, and disagreement then becomes a meaningful confidence signal.
- **Vehicle keypoints instead of a box.** A model that predicts wheel-contact points gives the true ground contact even when the roof is occluded and, unlike `y2`, does not move when the box is clipped.
- **Replace the assumed 50° FOV with a real homography.** Four points on the road plane with known spacing — lane markings, dashed-line pitch, a bridge span — pin the image-to-ground mapping. Metric distance stops depending on `horizontal_fov_deg` and on the assumed 1.82 m car width, and the negative-distance case disappears because extrapolation is bounded by the plane fit.
- **Estimate per-vehicle width instead of using a class constant.** The median observed width of a track, once mapped through the calibrated curve, is a better width prior for that specific vehicle than the class average, and it is available from data already collected.
- **Reject extrapolation explicitly.** Any frame whose bottom coordinate falls outside the calibrated bin range should produce `NaN`, not a value. That alone removes the physically invalid negative distance.

### 4. Making "closest" a measured quantity, not a chosen frame

- **Sub-frame refinement already exists** (parabola fit at the minimum) but is never validated. With ground truth, the parabolic estimate can be checked against interpolated labels to see whether it beats plain `argmin`.
- **Per-result confidence.** Combine estimator spread, detection confidence, track length, occlusion fraction, and zone membership into one score, then report the high-confidence subset separately. A system that says "these 40 results are solid, these 25 are not" is more useful than one that reports 202 rows of uniform apparent quality.
- **Ground truth is the precondition for all of it.** Twenty-five to thirty manually labelled vehicles are enough to turn every item above from an argument into a measurement: MAE in frames, % within ±1 and ±3 frames, ID-switch count, and a fragment-per-vehicle ratio. Without it, none of these changes can be shown to have helped.

### Expected Impact in Numbers

> **These are projections, not measurements.** No ground-truth file exists yet, so the "today" column mixes values measured on the dense 4K run with engineering estimates, and every "after" value is a *target to be checked* against manual labels — the number the change has to produce before it is called a success. They are written down precisely so that a rerun can falsify them.

The headline metric is **the share of vehicles whose selected frame falls within ±3 frames (±0.1 s) of a manual label.**

| Step | Closest frame within ±3 frames | Δ | Basis for the estimate |
|---|---:|---:|---|
| Today (stock YOLO11s + ByteTrack) | **~86%** (estimated) | — | On the sparse `highway.mp4` clip the estimators are stable and boxes are rarely occluded; the errors are the fragmented and boundary tracks. |
| \+ BoT-SORT with ReID and track stitching | **~91%** | +5 | Removes the duplicate-vehicle failure. Expected to cut the 202 reported tracks in the dense clip to roughly 110–130 and reduce ID switches by 30–50%. |
| \+ Boundary/extrapolation rejection and a higher `min_track_length` | **~93%** | +2 | Cheapest points on the list: these results are currently wrong by construction, not by estimation error. Costs recall — roughly 10–15% of rows move to "not reported" instead of "reported wrongly". |
| \+ Tiled inference at 4K (or `imgsz=1280`) | **~95%** | +2 | Recovers distant vehicles that today enter the track late. Small-object recall typically gains 10–20 points from slicing; the effect here is longer, earlier tracks rather than more vehicles. |
| \+ Independent monocular-depth vote | **~96–97%** | +1–2 | Breaks the shared failure mode of the three box-derived estimators. Gains are concentrated in the occluded cases, which are a minority of tracks. |
| \+ In-domain fine-tuning | **~97–98%** | +1 | Tighter boxes on this camera geometry. A ceiling effect applies: at 30 fps, ±3 frames is 0.1 s, and human annotators do not agree with each other much better than that. |

Secondary metrics, same caveat:

| Metric | Today | Target | Note |
|---|---:|---:|---|
| Reported tracks, dense 4K | 202 | 110–130 | Should approach the true number of distinct vehicles. |
| Estimator agreement within ±3 frames | 35.1% | 70–80% | Measured today. Rises mostly from tracking fixes, not from the estimators themselves. |
| Metric distance error | ±20–30% (est.) | ±5–10% | Only a homography with a known real-world scale gets there; the FOV assumption cannot. |
| Physically invalid results (negative distance) | 1 | 0 | Deterministic once extrapolation is rejected. |
| Tracks usable for reporting, dense 4K | 65 of 202 | 100–120 of ~120 | Fewer duplicates means a larger share of real vehicles clears the filters. |

The single most important line in this table is the first one: **~86% is an estimate, not a result.** Producing the ground-truth set is what converts the whole table from a plan into evidence, and it is why it sits first in the order of work below.

### Order of work

1. Ground-truth set — nothing else is measurable without it.
2. BoT-SORT + ReID and track stitching — largest single win, lowest effort.
3. Higher inference resolution or tiling on 4K.
4. Homography-based calibration, replacing the FOV assumption.
5. Independent monocular-depth estimator as a fourth vote.
6. In-domain fine-tuning, last, because it costs the most and only pays off once the tracking and geometry are no longer the bottleneck.

---

## If One Additional Hour Were Available

The first priority would be to create a small ground-truth set and improve tracking based on measurable errors. The hour would not be spent on additional UI or API features because the main risk is track reliability and the lack of external proof that the selected frame is correct.

### Proposed allocation

1. **20 minutes — manual labeling:** select 25–30 representative vehicles and label the closest frame for each one.

2. **25 minutes — filtering and tracking improvements:**

   - Report `entered_zone=true` tracks by default.
   - Reject tracks that begin or end at clip boundaries.
   - Reject negative distances and values outside the calibration range.
   - Increase the minimum track length.
   - Tune ByteTrack or compare it with BoT-SORT to reduce ID fragmentation.

3. **15 minutes — evaluation and rerun:** calculate errors against the manual labels and fix snapshot/contact-sheet generation.

This would make it possible to report a clear result such as “87% of vehicles were selected within three frames of the manual annotation,” instead of relying only on agreement between internal estimators.

---

## Summary

The system combines object detection, multi-object tracking, and simple perspective analysis. The central design decision was not to rely only on bounding-box size, but to combine several proximity cues and detect when a box no longer represents the complete vehicle.

The primary output is a frame number, timestamp, and bounding box for every vehicle. The annotated video, plots, and data files show how each decision was reached. A manually annotated ground-truth set would still be required before claiming a true external accuracy percentage.
