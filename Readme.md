# Camera Stream Processing POC

Real-time person detection and face recognition on camera streams. Connects to a **Synology Surveillance Station** NAS or any **ONVIF**-compatible camera, runs YOLOv8 person detection on each frame, optionally matches faces against a personal photo library exported from Adobe Lightroom Classic, and records annotated video clips.

Matched faces get **green** bounding boxes with the person's name. Unknown people get **yellow** "Person" boxes.

## Prerequisites

- Python 3.10 – 3.13
- A Synology NAS with Surveillance Station **or** any ONVIF-compatible IP camera
- Adobe Lightroom Classic catalog (`.lrcat`) with named faces (for face matching; optional)

## Quick Start

```bash
# 1. Clone and set up the environment
git clone <repo-url> && cd CameraStreamProcessingPOC
python -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # Linux/Mac
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env              # then edit with your camera details

# 3. Export faces and build encodings (optional — see below)

# 4. Run
python ReadFromStreamPoc2.py
```

## Step 1: Export Faces from Lightroom

`ExportLightroomFaces.py` reads your Lightroom Classic catalog, finds all confirmed face regions, crops them from the source photos, and saves them organized by person name. It then computes 512-d face encodings via `facenet-pytorch` for use in real-time matching.

### Export face crops and encodings

```bash
python ExportLightroomFaces.py \
    --catalog "C:/Users/you/Pictures/Lightroom/My Catalog.lrcat" \
    --output ./faces-output/
```

### Export face crops only (skip encodings)

```bash
python ExportLightroomFaces.py \
    --catalog "C:/Users/you/Pictures/Lightroom/My Catalog.lrcat" \
    --output ./faces-output/ \
    --skip-encodings
```

### Export options

| Flag | Default | Description |
|------|---------|-------------|
| `--catalog` | (required) | Path to the `.lrcat` file |
| `--output` | `./faces/` | Output directory |
| `--person` | all | Export only this person (exact name match) |
| `--padding` | 40 | Pixels of padding around each face crop |
| `--min-size` | 50 | Skip crops smaller than this in either dimension |
| `--skip-encodings` | off | Skip computing face encodings |
| `--include-suggested` | off | Include Lightroom's suggested (unconfirmed) faces |
| `--root-remap` | none | Remap catalog paths, format: `SRC=>DST` |

The `--root-remap` flag rewrites the path prefix stored in the catalog (e.g. `X:/Photos`) to where those files are accessible at runtime (e.g. `//nas/photos`). Specify it multiple times for multiple roots.

### Output structure

```
faces-output/
  John_Smith/
    face_1234.jpg
    face_5678.jpg
  Jane_Doe/
    face_9012.jpg
  encodings.pkl      # 512-d face encodings for all people
  export_log.json    # Export statistics
```

Face filenames use Lightroom's stable `face_id`, so re-running the export skips already-exported faces.

## Step 2: Configure the Environment

Create a `.env` file in the project root. Include whichever camera source(s) you intend to use:

```env
# Synology Surveillance Station
SYNOLOGY_IP=192.168.1.100
SYNOLOGY_PORT=5001
SYNOLOGY_USERNAME=your_username
SYNOLOGY_PASSWORD=your_password

# ONVIF camera (any ONVIF-compatible IP camera)
ONVIF_IP=192.168.1.50
ONVIF_PORT=80
ONVIF_USERNAME=admin
ONVIF_PASSWORD=your_password
```

You can also tune detection and matching settings at the top of `interfaces/watch_stream.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_ID` | 14 | Default Synology camera ID (override with `--camera`) |
| `DETECTION_CONFIDENCE` | 0.5 | Minimum YOLOv8 confidence for person detection |
| `RECORD_DURATION` | 30 | Seconds per recorded video clip |
| `ENCODINGS_PATH` | `./faces-output/encodings.pkl` | Path to the face encodings database |
| `MATCH_TOLERANCE` | 0.9 | Max face distance for a match (lower = stricter) |
| `MATCH_SKIP_FRAMES` | 5 | Attempt face matching every Nth frame with people |
| `MATCH_MIN_CONFIDENCE` | 0.5 | Ignore person detections below this confidence for matching |

## Step 3: Run the Application

```bash
# Synology — default camera
python ReadFromStreamPoc2.py

# Synology — specific camera ID
python ReadFromStreamPoc2.py --camera 5

# ONVIF camera — first media profile
python ReadFromStreamPoc2.py --source-type onvif

# ONVIF camera — specific media profile
python ReadFromStreamPoc2.py --source-type onvif --onvif-profile 1

# Video file (for testing)
python ReadFromStreamPoc2.py --source path/to/video.mp4

# Video file, looping
python ReadFromStreamPoc2.py --source path/to/video.mp4 --loop

# Disable saving recordings to disk
python ReadFromStreamPoc2.py --no-record
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | 14 | Synology camera ID |
| `--source` | none | Path to a video file (skips live camera) |
| `--loop` | off | Loop the source file when it ends |
| `--source-type` | `synology` | Live source: `synology` or `onvif` |
| `--onvif-profile` | 0 | ONVIF media profile index |
| `--no-record` | off | Disable saving annotated clips to `./recordings/` |

The application will:
1. Load YOLOv8 nano for person detection (auto-downloaded on first run)
2. Load face encodings from `encodings.pkl` (if available; skips face matching if not)
3. Connect to the camera and open the stream
4. Display a live window with annotated detections:
   - **Green** box + person name for recognized faces
   - **Yellow** "Person" box for unrecognized people
5. Record annotated video to `./recordings/` in 30-second clips (unless `--no-record`)

Press **q** in the display window or **Ctrl+C** in the terminal to stop.

### Running without face matching

Face matching is entirely optional. If `encodings.pkl` is missing or `facenet-pytorch` is not installed, the application prints a warning and runs person-detection only — all people get yellow "Person" boxes. No code changes required.

## How It Works

The detection system uses an event-driven pipeline so the frame loop never blocks on slow face encoding:

```
Frame --> PersonDetector --emits "person_detected"--> FaceMatcher --emits "face_matched"--> [any listener]
              |                                           |
              | (synchronous, ~15ms)                      | (async background thread, ~100ms)
              v                                           v
         Main loop draws                          Main loop polls latest results
         yellow "Person" boxes                    and upgrades to green "Name" boxes
```

- **PersonDetector** runs YOLOv8 nano on every frame and returns person bounding boxes immediately
- **FaceMatcher** picks up detection events on a background thread, runs `facenet-pytorch` against `encodings.pkl`, and publishes match results
- The main loop polls the latest match results each frame and associates them to current detections by bounding box proximity
- Frame skipping (`MATCH_SKIP_FRAMES`) bounds CPU usage by only running face matching every Nth frame containing people

## TODO

- **Low face detection rate in person crops** — MTCNN frequently fails to detect a face within the YOLO person bounding box (person too far away, facing away, partially occluded). Consider a wider/padded crop or a lower MTCNN confidence threshold.

- **Tolerance may be too loose** — `MATCH_TOLERANCE = 0.9` has produced suspected false positives. Try tightening to `0.7`–`0.8` and evaluate accuracy vs. miss rate.

- **Inconsistent matches across frames** — the same person can receive different name labels across nearby frames. Could be improved by temporal smoothing: accumulate results over a short window and report the majority-vote name.
