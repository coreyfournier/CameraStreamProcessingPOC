# Camera Stream Processing POC

Real-time person detection, face recognition, and activity logging for home surveillance cameras. Connects to **Synology Surveillance Station**, **ONVIF**-compatible cameras, or video files. Runs YOLOv8 person detection on each frame, matches faces against a known-faces database exported from Adobe Lightroom Classic, and logs all activity to a searchable database with a web UI.

- **Green** bounding boxes = recognized face with name
- **Yellow** bounding boxes = unrecognized person

## Architecture

The system runs as five services, orchestrated by Docker Compose:

```
                                      +------------+
  Cameras (up to 12)                  |  Frontend   |  :3000
        |                             |  (React)    |
        v                             +------+-----+
  +-----------+     +-------+                |
  |  Detector |---->| Redis |          +-----+------+
  |  (YOLOv8  |     |Streams|          | GraphQL API|  :8000
  |  + face   |     +---+---+          |  (FastAPI)  |
  |  matching)|         |              +-----+------+
  +-----------+         v                    |
                   +----------+        +-----+------+
                   | Consumer |------->|   SQLite    |
                   | (writer) |        |  (WAL mode) |
                   +----------+        +------------+
                        |
                   +----------+
                   |   Image  |
                   |  Storage |
                   | (filesystem)
                   +----------+
```

**Detector** reads camera streams, runs person detection (YOLOv8 nano) and face matching (facenet-pytorch), applies identity smoothing across frames, then publishes detections to Redis Streams.

**Consumer** reads from Redis Streams and writes detection records to SQLite (WAL mode, single writer). Also saves face/body crop images to the filesystem and runs periodic data retention cleanup.

**GraphQL API** serves detection data from SQLite and person images as static files.

**Frontend** is a React SPA for browsing, filtering, and managing detections.

## Prerequisites

- Python 3.10 -- 3.13
- Docker and Docker Compose (for the full system)
- A Synology NAS with Surveillance Station **or** any ONVIF-compatible IP camera **or** a video file for testing
- Adobe Lightroom Classic catalog (`.lrcat`) with named faces (optional, for face matching)

## Quick Start

### Option A: Docker Compose (recommended)

```bash
# 1. Clone and configure
git clone <repo-url> && cd CameraStreamProcessingPOC
cp .env.example .env              # edit with your camera credentials

# 2. Edit config.yaml with your cameras
#    Set redis.host to "redis" for Docker networking

# 3. Export faces (optional — see "Export Faces from Lightroom" below)

# 4. Start all services
docker compose up --build

# Frontend:  http://localhost:3000
# GraphQL:   http://localhost:8000/graphql
```

### Option B: Local development

```bash
# 1. Set up Python environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # Linux/Mac
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env              # edit with your camera credentials

# 3. Start Redis (required for logging)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 4. Run services in separate terminals:
python -m interfaces.watch_all_streams --config config.yaml   # detector
python -m interfaces.run_consumer --config config.yaml        # consumer
python -m interfaces.run_api --config config.yaml             # API server

# 5. Run frontend (optional)
cd frontend && npm install && npm run dev                     # http://localhost:5173
```

### Option C: Single camera (quick test)

No Redis or database needed — just detection and display:

```bash
# From a video file
python ReadFromStreamPoc2.py --source path/to/video.mp4 --loop --no-record

# From an ONVIF camera
python ReadFromStreamPoc2.py --source-type onvif

# From Synology
python ReadFromStreamPoc2.py --camera 14

# With a config file
python ReadFromStreamPoc2.py --config config.yaml --source path/to/video.mp4
```

## Configuration

All settings are centralized in `config.yaml`. Credentials use `${ENV_VAR}` syntax and are resolved from `.env`:

```yaml
cameras:
  - id: 1
    label: "front_door"
    source_type: "onvif"          # synology | onvif
  - id: 2
    label: "driveway"
    source_type: "synology"

synology:
  ip: "${SYNOLOGY_IP}"
  port: "${SYNOLOGY_PORT}"
  username: "${SYNOLOGY_USERNAME}"
  password: "${SYNOLOGY_PASSWORD}"
  secure: true
  cert_verify: true
  dsm_version: 7

detection:
  confidence_threshold: 0.5       # min YOLOv8 confidence for person detection
  encodings_path: "./faces-output/encodings.pkl"
  match_tolerance: 0.9            # max face distance for a match (lower = stricter)
  match_skip_frames: 5            # face match every Nth frame with people
  match_min_confidence: 0.5

smoothing:
  window_size: 10                 # frames in sliding window
  min_hit_ratio: 0.7              # 7/10 frames must agree on identity
  min_avg_confidence: 0.7         # avg confidence across window

redis:
  host: "localhost"               # use "redis" for Docker Compose
  port: 6379
  stream_name: "person_detections"

database:
  path: "./data/surveillance.db"

storage:
  person_images_dir: "./recordings/persons"

recording:
  enabled: true
  output_dir: "./recordings"
  clip_duration: 30               # seconds per video clip

retention:
  days: 30                        # auto-delete records older than this
  cleanup_interval_hours: 24

api:
  host: "0.0.0.0"
  port: 8000
```

### Environment variables (`.env`)

```env
# Synology Surveillance Station
SYNOLOGY_IP=192.168.1.100
SYNOLOGY_PORT=5001
SYNOLOGY_USERNAME=your_username
SYNOLOGY_PASSWORD=your_password

# ONVIF camera
ONVIF_IP=192.168.1.50
ONVIF_PORT=80
ONVIF_USERNAME=admin
ONVIF_PASSWORD=your_password
```

## Export Faces from Lightroom

`ExportLightroomFaces.py` extracts named faces from an Adobe Lightroom Classic catalog, crops them from source photos, and computes 512-d face encodings for real-time matching.

```bash
# Export all faces with encodings
python ExportLightroomFaces.py \
    --catalog "C:/Users/you/Pictures/Lightroom/My Catalog.lrcat" \
    --output ./faces-output/

# Export face crops only (skip encodings)
python ExportLightroomFaces.py \
    --catalog "path/to/catalog.lrcat" \
    --output ./faces-output/ \
    --skip-encodings

# Export a specific person
python ExportLightroomFaces.py \
    --catalog "path/to/catalog.lrcat" \
    --person "John Smith"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--catalog` | (required) | Path to the `.lrcat` file |
| `--output` | `./faces/` | Output directory |
| `--person` | all | Export only this person (exact name match) |
| `--padding` | 40 | Pixels of padding around each face crop |
| `--min-size` | 50 | Skip crops smaller than this in either dimension |
| `--skip-encodings` | off | Skip computing face encodings |
| `--include-suggested` | off | Include Lightroom's unconfirmed faces |
| `--root-remap` | none | Remap catalog paths, format: `SRC=>DST` |

Output structure:

```
faces-output/
  John_Smith/
    face_1234.jpg
    face_5678.jpg
  Jane_Doe/
    face_9012.jpg
  encodings.pkl      # 512-d face encodings for all people
  export_log.json    # export statistics
```

Face filenames use Lightroom's stable `face_id`, so re-running the export skips already-exported faces.

### Docker workflow (for dlib-dependent encodings)

dlib/face_recognition require native compilation that is difficult on Windows:

```bash
docker compose -f docker-compose.faces.yml build

CATALOG_DIR="C:\LightRoom" PHOTOS_DIR="/path/to/photos" OUTPUT_DIR="./faces-output" \
docker compose -f docker-compose.faces.yml run --rm export-faces \
    --catalog /catalog/catalog.lrcat --output /output \
    --root-remap "X:/=>/photos"
```

## How It Works

### Detection Pipeline

Each camera runs this pipeline per frame:

```
Frame --> PersonDetector --emits "person_detected"--> FaceMatcher --emits "face_matched"--> IdentitySmoother
              |                                           |                                       |
              | (synchronous, ~15ms)                      | (async background thread)              | (sliding window)
              v                                           v                                       v
         YOLOv8 nano                               facenet-pytorch                     Smoothed identity
         person boxes                              512-d encoding match                (requires N/M frames
                                                                                        to agree + min confidence)
```

1. **PersonDetector** (YOLOv8 nano) detects people in every frame, returning bounding boxes immediately.
2. **FaceMatcher** (facenet-pytorch) runs on a background thread, matching detected faces against `encodings.pkl`.
3. **IdentitySmoother** tracks people across frames using bounding box proximity and requires both a minimum hit ratio (e.g., 7/10 frames agree) and minimum average confidence before committing to an identity. This eliminates the flip-flopping problem.
4. Smoothed detections are published to **Redis Streams** with face/body crop images saved to the filesystem.
5. A single **Redis consumer** writes all detections to **SQLite** (WAL mode) — one writer avoids contention.

### Multi-Camera Support

The `MultiCameraOrchestrator` spawns one `CameraWorker` thread per camera. All workers share a single `RedisStreamProducer` (thread-safe) and `PersonImageStorage` (unique detection IDs). Designed for at least 12 simultaneous streams.

### Unknown Person Clustering

Unidentified people are periodically clustered by face encoding similarity (L2 distance with configurable threshold). When a cluster's centroid matches a known person from `encodings.pkl`, the system suggests the identity. A user must confirm via the UI before past records are updated.

### Data Retention

A background job runs every `retention.cleanup_interval_hours` and deletes detection records and associated images older than `retention.days`. Images are deleted before database records.

## CLI Reference

### Single camera

```bash
python ReadFromStreamPoc2.py [options]
# or
python -m interfaces.watch_stream [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | none | Path to YAML config file |
| `--camera` | 14 | Synology camera ID |
| `--source` | none | Path to a video file (skips live camera) |
| `--loop` | off | Loop the source file when it ends |
| `--source-type` | `synology` | Live source: `synology` or `onvif` |
| `--onvif-profile` | 0 | ONVIF media profile index |
| `--no-record` | off | Disable saving annotated clips |

### All cameras

```bash
python -m interfaces.watch_all_streams --config config.yaml
```

### Redis consumer

```bash
python -m interfaces.run_consumer --config config.yaml
```

### GraphQL API server

```bash
python -m interfaces.run_api --config config.yaml
```

### Manual data cleanup

```bash
python -m interfaces.run_cleanup --config config.yaml [--days 30]
```

## Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| `redis` | — | Redis 7 message broker (internal) |
| `detector` | — | Camera processing + person detection |
| `consumer` | — | Redis-to-SQLite writer + retention cleanup |
| `api` | 8000 | GraphQL API + image serving |
| `frontend` | 3000 | React web UI |

```bash
docker compose up --build        # start all services
docker compose up -d             # start in background
docker compose logs -f detector  # follow detector logs
docker compose down              # stop all services
```

Shared volumes:
- `recordings` — video clips + person crop images (detector writes, API serves)
- `db-data` — SQLite database (consumer writes, API reads)
- `redis-data` — Redis persistence

## Web UI

The frontend at `http://localhost:3000` provides:

- **Dashboard** — recent activity feed with auto-refresh (every 10s)
- **Person Detail** — all appearances of a specific person, filterable by date
- **Unknown Clusters** — review system-suggested identity matches, confirm or reject

## Project Structure

```
domain/                              # Pure value objects
  detection/events.py                # FrameContext, PersonDetection, FaceMatchResult,
                                     # SmoothedIdentity, PersonLogEntry
shared/
  event_emitter.py                   # Thread-safe EventEmitter

infrastructure/                      # External integrations
  config.py                          # YAML config loader with ${ENV_VAR} resolution
  camera/
    synology_camera_source.py        # Synology Surveillance Station client
    onvif_camera_source.py           # ONVIF camera client
    opencv_frame_reader.py           # Background thread frame reader
  detection/
    yolo_person_detector.py          # YOLOv8 nano person detector
    facenet_face_matcher.py          # facenet-pytorch face matcher
  recording/
    avi_clip_writer.py               # Auto-rotating AVI clip writer
  database/
    person_log_db.py                 # SQLite database (WAL mode)
    schema.sql                       # Database schema
  messaging/
    redis_stream_producer.py         # Redis Streams publisher
    redis_stream_consumer.py         # Redis Streams consumer
  storage/
    person_image_storage.py          # Filesystem image storage

application/                         # Orchestration
  detection_pipeline.py              # DetectionPipeline + AsyncFaceMatcherWrapper
  detection_smoother.py              # IdentitySmoother (temporal smoothing)
  stream_processor.py                # Frame annotation (draw bounding boxes)
  camera_worker.py                   # Single-camera processing loop
  multi_camera_orchestrator.py       # Multi-camera manager (12+ streams)
  unknown_person_clusterer.py        # Face encoding clustering
  retention_manager.py               # Data retention cleanup
  catalog_exporter.py                # Lightroom face export functions

interfaces/                          # Entry points
  watch_stream.py                    # Single-camera CLI
  watch_all_streams.py               # Multi-camera CLI
  run_consumer.py                    # Redis consumer CLI
  run_api.py                         # GraphQL API server CLI
  run_cleanup.py                     # Manual cleanup CLI
  export_faces.py                    # Lightroom face export CLI
  api/
    schema.py                        # GraphQL types, queries, mutations
    server.py                        # FastAPI app factory

frontend/                            # React + Vite SPA
  src/
    pages/Dashboard.jsx
    pages/PersonDetail.jsx
    pages/UnknownClusters.jsx
    components/DetectionCard.jsx
    components/DateFilter.jsx
    components/Layout.jsx
    graphqlClient.js

ReadFromStreamPoc2.py                # Backward-compat shim -> interfaces/watch_stream.py
ExportLightroomFaces.py              # Backward-compat shim -> interfaces/export_faces.py
```

## Running Without Face Matching

Face matching is optional. If `encodings.pkl` is missing or `facenet-pytorch` is not installed, the application runs person-detection only (all people get yellow "Person" boxes). No code changes required.

## Known Limitations

- **Low face detection rate in person crops** -- MTCNN can fail to detect faces within YOLO person bounding boxes when people are far away, facing away, or partially occluded.
- **GPU recommended for 12 cameras** -- Running YOLOv8 on 12 simultaneous streams is CPU-intensive. A CUDA-capable GPU significantly improves throughput.
- **SQLite single-writer** -- The Redis Streams buffer mitigates this, but under extreme load (12+ cameras with many detections), evaluate switching to PostgreSQL.
