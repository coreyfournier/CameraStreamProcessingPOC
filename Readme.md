# Camera Stream Processing POC

Real-time person detection and face recognition on Synology Surveillance Station camera streams. Detects people using MobileNet-SSD, then matches their faces against a personal photo library exported from Adobe Lightroom Classic.

Matched faces get **green** bounding boxes with the person's name. Unknown people get **yellow** "Person" boxes.

## Prerequisites

- Python 3.10 - 3.13
- A Synology NAS with Surveillance Station and at least one camera configured
- Adobe Lightroom Classic catalog (`.lrcat`) with named faces (for face matching)
- Docker (recommended for building face encodings on Windows, since dlib is difficult to compile natively)

## Quick Start

```bash
# 1. Clone and set up the environment
git clone <repo-url> && cd CameraStreamProcessingPOC
python -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # Linux/Mac
pip install -r requirements.txt

# 2. Configure Synology credentials
cp .env.example .env              # then edit with your NAS details

# 3. Export faces and build encodings (see detailed steps below)

# 4. Run
python ReadFromStreamPoc2.py
```

## Step 1: Export Faces from Lightroom

`ExportLightroomFaces.py` reads your Lightroom Classic catalog's SQLite database, finds all named/confirmed face regions, crops them from the source photos, and saves them organized by person.

### Option A: Export face crops only (no dlib required)

If you just want the cropped face images without encodings (useful for inspection or if you'll compute encodings separately):

```bash
python ExportLightroomFaces.py \
    --catalog "C:/Users/you/Pictures/Lightroom/My Catalog.lrcat" \
    --output ./faces/ \
    --skip-encodings
```

### Option B: Export with encodings via Docker (recommended on Windows)

dlib and `face_recognition` require native C++ compilation that is painful on Windows. The provided Docker image handles this.

```bash
# Build the image (one-time, ~3 min for dlib compilation)
docker compose -f docker-compose.faces.yml build

# Run the export
# - CATALOG_DIR: directory containing your .lrcat file
# - PHOTOS_DIR:  root of your source photos
# - OUTPUT_DIR:  where face crops and encodings.pkl will be saved
CATALOG_DIR="C:/LightRoom" \
PHOTOS_DIR="//nas/photos" \
OUTPUT_DIR="./faces" \
docker compose -f docker-compose.faces.yml run --rm export-faces \
    --catalog /catalog/"My Catalog.lrcat" \
    --output /output \
    --root-remap "X:/=>/photos"
```

The `--root-remap` flag is key: Lightroom stores absolute Windows paths (e.g. `X:/2022/vacation/photo.jpg`) in the catalog. The flag rewrites the prefix so the container can find the photos at its mount point (e.g. `/photos/2022/vacation/photo.jpg`). Specify it multiple times if your photos span multiple roots.

### Option C: Export with encodings natively (Linux/Mac)

If you can install dlib on your system:

```bash
pip install dlib face_recognition

python ExportLightroomFaces.py \
    --catalog "/path/to/My Catalog.lrcat" \
    --output ./faces/
```

### Export options

| Flag | Default | Description |
|------|---------|-------------|
| `--catalog` | (required) | Path to the `.lrcat` file |
| `--output` | `./faces/` | Output directory |
| `--person` | all | Export only this person (exact name match) |
| `--padding` | 40 | Pixels of padding around each face crop |
| `--min-size` | 50 | Skip crops smaller than this in either dimension |
| `--skip-encodings` | false | Skip computing face encodings |
| `--root-remap` | none | Remap catalog paths, format: `SRC=>DST` |

### Output structure

```
faces/
  John_Smith/
    face_1234.jpg
    face_5678.jpg
  Jane_Doe/
    face_9012.jpg
  encodings.pkl          # 128-d face encodings for all people
  export_log.json        # Export statistics
```

Face filenames use Lightroom's stable `face_id`, so re-running the export skips already-exported faces.

## Step 2: Build Face Encodings

If you exported with `--skip-encodings` in Step 1, you need to generate `encodings.pkl` before face matching will work. Re-run the export without that flag (Docker method recommended):

```bash
CATALOG_DIR="C:/LightRoom" \
PHOTOS_DIR="//nas/photos" \
OUTPUT_DIR="./faces" \
docker compose -f docker-compose.faces.yml run --rm export-faces \
    --catalog /catalog/"My Catalog.lrcat" \
    --output /output \
    --root-remap "X:/=>/photos"
```

The encoding step walks every `face_*.jpg` in the output directory, computes a 128-dimensional face encoding via `face_recognition`, and writes them all to `encodings.pkl`. This file is what the real-time matcher loads at startup.

## Step 3: Configure the Environment

Create a `.env` file with your Synology NAS credentials:

```env
SYNOLOGY_IP=192.168.1.100
SYNOLOGY_PORT=5001
SYNOLOGY_USERNAME=your_username
SYNOLOGY_PASSWORD=your_password
```

You can also tune detection and matching settings at the top of `ReadFromStreamPoc2.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_ID` | 2 | Default camera ID (override with `--camera`) |
| `DETECTION_CONFIDENCE` | 0.5 | Minimum MobileNet-SSD confidence for person detection |
| `RECORD_DURATION` | 30 | Seconds per recorded video clip |
| `ENCODINGS_PATH` | `./faces/encodings.pkl` | Path to the face encodings database |
| `MATCH_TOLERANCE` | 0.6 | Max face distance for a match (lower = stricter) |
| `MATCH_SKIP_FRAMES` | 5 | Attempt face matching every Nth frame with people |
| `MATCH_MIN_CONFIDENCE` | 0.5 | Ignore person detections below this confidence for matching |

## Step 4: Run the Application

```bash
# Watch the default camera (ID 2)
python ReadFromStreamPoc2.py

# Watch a specific camera by ID
python ReadFromStreamPoc2.py --camera 5
```

The application will:
1. Load the MobileNet-SSD person detection model
2. Load face encodings from `encodings.pkl` (if available; warns and continues without if not)
3. Connect to your Synology NAS and open the camera stream
4. Display a live window with annotated detections:
   - **Green** box + person name for recognized faces
   - **Yellow** box + "Person" for detected but unrecognized people
5. Record annotated video to `./recordings/` in 30-second clips

Press **q** in the display window or **Ctrl+C** in the terminal to stop.

### Running without face matching

Face matching is entirely optional. If `face_recognition`/dlib are not installed or `encodings.pkl` is missing, the application prints a warning and runs person detection only (all people get yellow "Person" boxes). No code changes needed.

## How It Works

The detection system uses an event-driven pipeline so components are decoupled and the frame loop never blocks on slow face encoding:

```
Frame --> PersonDetector --emits "person_detected"--> FaceMatcher --emits "face_matched"--> [any listener]
              |                                           |
              | (synchronous ~15ms)                       | (async background thread ~100ms)
              v                                           v
         Main loop draws                          Main loop polls latest results
         yellow "Person" boxes                    and upgrades to green "Name" boxes
```

- **PersonDetector** runs MobileNet-SSD on every frame and returns person bounding boxes immediately
- **FaceMatcher** picks up detection events on a background thread, runs `face_recognition` against `encodings.pkl`, and publishes match results
- The main loop polls the latest match results each frame and associates them to current detections by bounding box proximity (since async results may be from a slightly earlier frame)
- Frame skipping ensures the matcher only processes every Nth frame with people to bound CPU usage

## Alternative: Direct RTSP (no Synology)

For testing against any RTSP camera source without Synology integration:

```bash
python ReadFromStreamPoc1.py \
    --rtsp rtsp://camera-ip:554/stream \
    --proto MobileNetSSN/MobileNetSSD_deploy.prototxt \
    --model MobileNetSSN/MobileNetSSD_deploy.caffemodel \
    --out output.mp4
```

Note: this alternative script does not include the event-driven pipeline or face matching.

## TODO

Issues observed during testing:

- **Low face detection rate in person crops** — MTCNN frequently fails to detect a face within the YOLO person bounding box (person too far from camera, facing away, partially occluded). When MTCNN finds no face, the whole crop is resized as a fallback, which produces poor embeddings. Consider using a wider/padded crop region or a lighter face detector with a lower confidence threshold.

- **Tolerance may be too loose** — `MATCH_TOLERANCE = 0.9` produced suspected false positives (e.g. Eli Ellsworth matched when the person's face was not clearly visible). Try tightening to `0.7`–`0.8` and evaluate accuracy vs. miss rate.

- **Inconsistent matches across frames** — the same person receives different name labels across nearby frames (e.g. Kate Fournier → Keith Fournier → Eli Ellsworth within seconds). Could be improved by temporal smoothing: accumulate match results over a short window and report the majority-vote name.

- **README is outdated** — still references MobileNet-SSD, dlib, `face_recognition`, 128-d encodings, and Docker as the recommended encoding workflow. All of these have been replaced by YOLOv8 and `facenet-pytorch`; the README needs a full update to reflect the current stack.
