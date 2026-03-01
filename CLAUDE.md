# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python proof-of-concept for real-time camera stream processing with object detection. Connects to a Synology Surveillance Station NAS via its API, reads RTSP/MJPEG camera streams, runs YOLOv8 person detection on each frame, optionally matches faces against a known-faces database, and records annotated video clips.

## Setup and Running

```bash
# Create and activate virtual environment (OpenCV requires Python < 3.14)
python -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # Linux/Mac

# Install dependencies
python -m pip install -r requirements.txt

# Run main application (connects to Synology, processes stream, records clips)
python ReadFromStreamPoc2.py

# Run CLI-based alternative (direct RTSP, no Synology integration)
python ReadFromStreamPoc1.py --rtsp <url> --proto MobileNetSSN/MobileNetSSD_deploy.prototxt --model MobileNetSSN/MobileNetSSD_deploy.caffemodel --out output.mp4
```

There are no tests, linting, or CI/CD configured.

## Architecture

The project is organized into DDD-style layers:

```
domain/          — Pure value objects, no dependencies on infrastructure
shared/          — Cross-cutting utilities (EventEmitter)
infrastructure/  — External integrations (camera, detection models, recording)
application/     — Orchestration and use-case logic
interfaces/      — CLI entry points (main() + parse_args())
```

Root `ReadFromStreamPoc2.py` and `ExportLightroomFaces.py` are 3-line shims for backward compatibility.

### Entry points

**ReadFromStreamPoc2.py** → **interfaces/watch_stream.py** — Main entry point. Connects to Synology via `CameraSource` (or reads a video file via `--source`), reads frames in a loop, runs person detection and optional face matching via `DetectionPipeline`, draws annotated bounding boxes (green for matched faces, yellow for unknown people), and writes output to AVI clips (30-second segments) in `./recordings/`.

**ExportLightroomFaces.py** → **interfaces/export_faces.py** — Standalone CLI tool that extracts named faces from an Adobe Lightroom Classic catalog (`.lrcat` SQLite database). Crops face regions from source photos (handling EXIF orientation), saves them organized by person name, and optionally computes 512-d face encodings via `facenet-pytorch` for real-time matching.

```bash
# Export all faces (without encodings — no dlib needed)
python ExportLightroomFaces.py --catalog "path/to/My Catalog.lrcat" --output ./faces/ --skip-encodings

# Export a specific person with encodings
python ExportLightroomFaces.py --catalog "path/to/My Catalog.lrcat" --person "John Smith"

# Full options
python ExportLightroomFaces.py --catalog <path> [--output ./faces/] [--person "Name"] [--padding 40] [--min-size 50] [--skip-encodings] [--root-remap "SRC=>DST"]
```

Output structure: `faces/<Person_Name>/face_<face_id>.jpg`, `faces/encodings.pkl`, `faces/export_log.json`. Face filenames use Lightroom's stable `face_id` so subsequent runs skip already-exported faces.

### Event-driven detection pipeline

The detection system uses an observer/listener pattern for decoupled, extensible processing:

```
Frame → PersonDetector ──emits "person_detected"──→ FaceMatcher ──emits "face_matched"──→ [any listener]
              │                                           │
              │ (returns detections synchronously)        │ (runs on background thread)
              ↓                                           ↓
         Main loop draws                          Main loop polls latest results
         yellow "Person" boxes                    and upgrades to green "Name" boxes
```

### Key files

**domain/detection/events.py** — Dataclasses: `FrameContext`, `PersonDetection`, `PersonDetectionEvent`, `FaceMatchResult`, `FaceMatchEvent`.

**shared/event_emitter.py** — Thread-safe `EventEmitter` with `on()`/`off()`/`emit()`.

**infrastructure/detection/yolo_person_detector.py** — `PersonDetector`: wraps YOLOv8 nano (`yolov8n.pt`, auto-downloaded on first run). Filters to COCO class 0 (person) only. `process_frame()` returns detections synchronously AND emits `"person_detected"` events with cropped person images.

**infrastructure/detection/facenet_face_matcher.py** — `FaceMatcher`: listens for person detections, runs `facenet-pytorch` to match against `encodings.pkl`. Emits `"face_matched"` events. Gracefully degrades if facenet-pytorch is not installed (becomes a no-op).

**infrastructure/camera/synology_camera_source.py** — `CameraSource`: Synology Surveillance Station client wrapper. Handles connection, camera enumeration, stream URL retrieval, and URL fixup.

**infrastructure/camera/opencv_frame_reader.py** — `OpenCVFrameReader`: reads RTSP frames on a background thread, retaining only the most recent frame.

**infrastructure/recording/avi_clip_writer.py** — `AviClipWriter`: encapsulates OpenCV VideoWriter, auto-rotates clips every N seconds, generates timestamped filenames.

**application/detection_pipeline.py** — `AsyncFaceMatcherWrapper` (background thread, bounded queue, frame skipping) and `DetectionPipeline` that wires everything together. Exposes `process_frame()` (synchronous), `get_latest_matches()` (non-blocking poll), and `on_face_matched()` (register external listeners).

**application/stream_processor.py** — `draw_detections()`, `_associate_matches()`, and box-geometry helpers.

**application/catalog_exporter.py** — All face export functions: `open_catalog()`, `query_named_faces()`, `resolve_image_path()`, `load_image_with_orientation()`, `crop_face()`, `sanitize_name()`, `compute_encodings()`, `save_export_log()`.

**ReadFromStreamPoc1.py** — Simpler CLI alternative that takes an RTSP URL directly (no Synology integration). Useful for testing detection against any RTSP source.

**MobileNetSSN/** — Pre-trained MobileNet-SSD Caffe model files (prototxt + caffemodel). Used only by `ReadFromStreamPoc1.py` (the legacy RTSP script). The main pipeline (`ReadFromStreamPoc2.py`) uses YOLOv8 instead.

### Docker workflow (recommended for face encodings)

dlib/face_recognition require native compilation that is difficult on Windows. A multi-stage Docker build (`Dockerfile.faces`) provides Python 3.12 with pre-compiled dlib. Use `--root-remap` to translate Lightroom's Windows absolute paths to container mount points.

```bash
# Build the image (~3 min first time for dlib compilation)
docker compose -f docker-compose.faces.yml build

# Run with bind-mount volumes (generic)
CATALOG_DIR="C:\LightRoom" PHOTOS_DIR="/path/to/photos" OUTPUT_DIR="./faces-output" \
docker compose -f docker-compose.faces.yml run --rm export-faces \
    --catalog /catalog/catalog.lrcat --output /output \
    --root-remap "X:/=>/photos"

# Run with local compose override for NAS CIFS mounts
# (create docker-compose.faces.local.yml — gitignored — with CIFS volume config)
docker compose -f docker-compose.faces.yml -f docker-compose.faces.local.yml run --rm export-faces \
    --catalog /catalog/catalog.lrcat --output /output \
    --root-remap "X:/=>/photos"
```

Docker volume mounts:
- `/catalog` (ro) — directory containing the `.lrcat` file (and WAL/SHM files)
- `/photos` (ro) — source photos root (mapped from `PHOTOS_DIR` or a CIFS volume)
- `/output` — face crops and `encodings.pkl` output

The `--root-remap` flag translates catalog paths (e.g. `X:/2022/photo.jpg`) to container paths (e.g. `/photos/2022/photo.jpg`). Can be specified multiple times for multiple roots.

## Environment Configuration

Synology credentials are loaded from `.env` (not committed):
- `ip_address` — NAS hostname/IP
- `port` — Surveillance Station port (5001 for HTTPS)
- `username` / `password` — NAS credentials

Hardcoded settings in `interfaces/watch_stream.py`: `CAMERA_ID`, `DETECTION_CONFIDENCE`, `RECORD_DURATION`, `ENCODINGS_PATH`, `MATCH_SKIP_FRAMES`, `MATCH_TOLERANCE`, `MATCH_MIN_CONFIDENCE`, SSL options (`secure`, `cert_verify`, `dsm_version`).
