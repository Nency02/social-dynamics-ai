# Social Dynamics AI

Real-time AI system to analyze group interactions using pose detection and social behavior modeling.

## Project Goal

Build an end-to-end pipeline that takes camera frames, detects people and pose keypoints, then infers social interaction roles such as dominant, engaged, and peripheral.

Target deployment device: NVIDIA Jetson Nano 2GB (after desktop/laptop development is stable).

## Current Progress (Checked from Code)

Implemented now:

- Real-time camera capture loop in `main.py`
- YOLOv8 pose inference via `src/vision/pose.py`
- Per-person keypoint extraction in `src/vision/keypoints.py`
- Orientation feature (shoulder-angle) calculation in `src/vision/keypoints.py`
- JSON export of per-frame results to `outputs/keypoints.json`
- On-screen annotated pose preview with OpenCV

Partially implemented / not yet production-ready:

- `src/vision/detect.py` exists but uses `yolov8n.pt` (generic detection), not integrated into main flow
- No identity tracking across frames yet (person IDs reset per frame)
- No behavior/social scoring module yet (`src/behavior` not created)
- No role classification module yet
- No visualization overlay for social labels yet
- No evaluation scripts, tests, or performance benchmarks yet

## Architecture

```text
Camera
	-> Pose Detection (YOLOv8 Pose)
	-> Keypoint Extraction
	-> Behavior Feature Extraction
	-> Interaction Scoring
	-> Role Classification
	-> Visualization Overlay
```

Ownership split:

- Person 1 (Vision Engineer, stronger GPU laptop): camera + detection + pose + tracking + inference optimization
- Person 2 (Social Intelligence Engineer, 8 GB laptop): behavior features + scoring + role logic + social visualization

## Why This Dependency Order

Person 2 depends on clean keypoint output. Because of that:

1. Person 1 must first stabilize keypoint schema and frame outputs.
2. Person 2 can start algorithm R and D in parallel using recorded keypoint JSON.
3. Integration starts only after both API contracts are stable.

This prevents blocking and rework.

## Suggested Repository Structure

```text
social-dynamics-ai/
	main.py
	requirements.txt
	README.md
	outputs/
		keypoints.json
	src/
		vision/
			camera.py
			detect.py
			pose.py
			keypoints.py
			tracking.py                 # to add
		behavior/
			features.py                 # to add
			scoring.py                  # to add
			roles.py                    # to add
		visualization/
			overlay.py                  # to add
```

## Team Workflow (Step by Step)

### Phase 0: Alignment and Contract (Both, Day 0)

1. Freeze output schema for pose frames (must be shared contract).
2. Agree thresholds and metric definitions (engagement, dominance, cohesion).
3. Create branches and naming rules.

Deliverable:

- API contract document section in this README.

### Phase 1: Vision Stability (Person 1 starts first, Day 1-2)

1. Stabilize camera and pose inference loop.
2. Standardize per-frame JSON format.
3. Add tracking (`track_id`) to keep person identity across frames.
4. Save short sample recordings (JSON + optional video) for Person 2.

Deliverable:

- Reliable stream of keypoints with `track_id`, confidence, and timestamps.

Why first:

- Behavior logic needs stable person identity and keypoints.

### Phase 2: Social Intelligence Core (Person 2 parallel start after sample data, Day 2-4)

1. Build behavior feature extraction from keypoints:
	 - orientation
	 - gesture activity
	 - distance/proximity
	 - facing relation between people
2. Build scoring functions:
	 - engagement score
	 - dominance score
	 - participation score
3. Build role assignment logic:
	 - dominant
	 - engaged
	 - peripheral
4. Build group-level metric:
	 - cohesion score

Deliverable:

- `behavior` module that takes keypoint payload and returns social metrics per person.

Why now:

- Runs on lightweight compute, no GPU needed.

### Phase 3: Visualization Integration (Both, Day 5)

1. Overlay social role labels on frame.
2. Draw score bars and group cohesion indicator.
3. Add color-coded role display.

Deliverable:

- Live output frame with pose + social labels.

### Phase 4: End-to-End Integration (Both, Day 6)

1. Merge `vision` + `behavior` + `visualization` into `main.py` pipeline.
2. Fix interface mismatches.
3. Run 3 scenario demos:
	 - small discussion
	 - single dominant speaker
	 - low engagement group

Deliverable:

- One-command local demo run.

### Phase 5: Jetson Deployment and Optimization (Both, Day 7)

1. Move final pipeline to Jetson Nano.
2. Reduce model/image size if needed (speed optimization).
3. Tune FPS and resolution trade-offs.

Deliverable:

- Demo-ready edge deployment.

## API Contract (Person 1 -> Person 2)

Each frame should provide:

```json
{
	"frame_id": 101,
	"timestamp": 1773920204.925,
	"people": [
		{
			"track_id": 7,
			"bbox_xyxy": [x1, y1, x2, y2],
			"confidence": 0.83,
			"keypoints": [[x, y], ...],
			"keypoint_confidence": [0.98, ...],
			"orientation_angle_deg": -12.4
		}
	]
}
```

Notes:

- `track_id` should be stable across frames.
- Keypoint order must remain consistent with YOLOv8 pose keypoint indexing.

## Git Workflow

Branch strategy:

- `main`: stable integration
- `feature/vision-*`: Person 1 tasks
- `feature/behavior-*`: Person 2 tasks
- `feature/visualization-*`: shared overlay work

Daily routine:

1. Pull latest `main`.
2. Work on feature branch.
3. Push and open PR.
4. Teammate reviews and merges.

Integration rule:

- Merge only when sample input/output contract tests pass.

## Milestone Timeline

1. Day 1-2: Vision pipeline stabilized and sample keypoints exported.
2. Day 2-4: Behavior scoring and role classifier implemented.
3. Day 5: Visualization overlay completed.
4. Day 6: End-to-end pipeline merged and validated.
5. Day 7: Jetson deployment and final demo polish.

## Success Criteria

Project is demo-ready when:

- Pose keypoints are extracted in real time.
- Each person has stable track ID over time.
- Engagement/dominance/cohesion metrics are computed from keypoints.
- Roles are displayed live on video.
- Pipeline runs on Jetson with acceptable FPS for demo.

## Quick Start (Current State)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

Current output:

- Live pose visualization window
- Latest frame data written to `outputs/keypoints.json`
