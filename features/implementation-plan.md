# Surveillance Nation — Implementation Plan

This plan implements the features described in `survellance-nation.md`. Phases are ordered by dependency — each phase builds on the previous.

---

## Phase 0: Foundation — Configuration & Refactor

**Goal:** Extract the stream processing into a reusable module and introduce centralized configuration to support multi-camera setups.

### 0.1 Centralized configuration

Replace hardcoded constants in `interfaces/watch_stream.py` with a YAML config file.

- Create `config.yaml` schema:
  ```yaml
  cameras:
    - id: 1
      label: "front_door"
      source_type: "onvif"       # synology | onvif
      onvif_ip: "192.168.1.50"
      onvif_port: 80
      onvif_username: "${ONVIF_USERNAME}"
      onvif_password: "${ONVIF_PASSWORD}"
    - id: 2
      label: "driveway"
      source_type: "synology"
      synology_camera_id: 14

  synology:                       # shared Synology connection
    ip: "${SYNOLOGY_IP}"
    port: "${SYNOLOGY_PORT}"
    username: "${SYNOLOGY_USERNAME}"
    password: "${SYNOLOGY_PASSWORD}"

  detection:
    confidence_threshold: 0.5
    encodings_path: "./faces-output/encodings.pkl"
    match_tolerance: 0.9
    match_skip_frames: 5
    match_min_confidence: 0.5

  smoothing:
    window_size: 10               # number of frames in sliding window
    min_hit_ratio: 0.7            # 7/10 frames must agree
    min_avg_confidence: 0.7       # average confidence across window

  redis:
    host: "localhost"
    port: 6379

  database:
    path: "./surveillance.db"

  storage:
    person_images_dir: "./recordings/persons"

  recording:
    enabled: true
    output_dir: "./recordings"
    clip_duration: 30
    fps: 20

  retention:
    days: 30
    cleanup_interval_hours: 24

  api:
    host: "0.0.0.0"
    port: 8000
  ```

- New file: `infrastructure/config.py` — load YAML, resolve `${ENV_VAR}` references, validate, expose as typed dataclass/dict.
- Update `interfaces/watch_stream.py` to read from config instead of hardcoded constants.

### 0.2 Refactor stream processing into a standalone module

- Move the core frame-processing loop out of `interfaces/watch_stream.py` into `application/camera_worker.py`.
- `CameraWorker` class encapsulates:
  - Opening a camera source (Synology or ONVIF) by config.
  - Running the `DetectionPipeline` for that camera.
  - Reading frames via `OpenCVFrameReader`.
  - Optionally recording clips via `AviClipWriter`.
- `interfaces/watch_stream.py` becomes a thin CLI that instantiates one `CameraWorker` and runs it (preserving current single-camera behavior).

**Files created/modified:**
| File | Action |
|------|--------|
| `config.yaml` | Create |
| `infrastructure/config.py` | Create |
| `application/camera_worker.py` | Create |
| `interfaces/watch_stream.py` | Modify (use config + CameraWorker) |

---

## Phase 1: Person Recognition Smoothing

**Goal:** Eliminate identity flip-flopping by requiring consistent identification across a sliding window of frames.

### 1.1 Detection smoother

- New file: `application/detection_smoother.py`
- `IdentitySmoother` class:
  - Maintains a per-tracked-person sliding window (last N frames).
  - Uses bounding box proximity (IoU or center distance) to track the same person across frames.
  - For each tracked person, stores recent `(name, confidence)` tuples.
  - `smooth(detections, match_results) -> list[SmoothedIdentity]`:
    - Compute hit ratio: `count(name == top_name) / window_size`
    - Compute average confidence for top name.
    - Only emit identity if both thresholds are met.
    - Otherwise, label as "Unknown".
  - Configurable: `window_size`, `min_hit_ratio`, `min_avg_confidence`.

### 1.2 Integrate into pipeline

- Update `application/detection_pipeline.py` to run `IdentitySmoother` after face matching.
- `DetectionPipeline.get_latest_matches()` returns smoothed results.
- Update `application/stream_processor.py` to draw smoothed labels.

**Files created/modified:**
| File | Action |
|------|--------|
| `application/detection_smoother.py` | Create |
| `application/detection_pipeline.py` | Modify |
| `application/stream_processor.py` | Modify (use smoothed labels) |

---

## Phase 2: Redis Streams & SQLite Logging

**Goal:** Publish person detections to Redis Streams; consume and persist to SQLite.

### 2.1 Domain events for logging

- Extend `domain/detection/events.py` with:
  - `PersonLogEntry` dataclass: `detection_id`, `timestamp`, `camera_id`, `camera_label`, `person_name` (nullable), `confidence`, `face_crop_path`, `body_crop_path`, `face_encoding` (for clustering).

### 2.2 Person image storage

- New file: `infrastructure/storage/person_image_storage.py`
- `PersonImageStorage` class:
  - `save(detection_id, face_crop, body_crop, timestamp) -> (face_path, body_path)`
  - Path convention: `{base_dir}/YYYY/MM/DD/{detection_id}_face.jpg`
  - `delete_before(cutoff_date)` — bulk delete old images for retention.

### 2.3 Redis Streams producer

- New file: `infrastructure/messaging/redis_stream_producer.py`
- `RedisStreamProducer` class:
  - `publish(person_log_entry: PersonLogEntry)` — serialize and `XADD` to Redis stream.
  - Saves face/body crops via `PersonImageStorage` before publishing (so paths are in the message).
- Integrate into `CameraWorker` or `DetectionPipeline`: after detection + smoothing, publish to Redis.

### 2.4 SQLite database

- New file: `infrastructure/database/schema.sql`
  ```sql
  CREATE TABLE IF NOT EXISTS person_detections (
      detection_id TEXT PRIMARY KEY,
      timestamp    TEXT NOT NULL,           -- ISO 8601
      camera_id    INTEGER NOT NULL,
      camera_label TEXT NOT NULL,
      person_name  TEXT,                    -- NULL if unidentified
      confidence   REAL,
      face_crop_path TEXT,
      body_crop_path TEXT,
      face_encoding  BLOB,                 -- 512-d float32 for clustering
      cluster_id     INTEGER,              -- FK to unknown_clusters (nullable)
      created_at     TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE INDEX idx_detections_person ON person_detections(person_name, timestamp);
  CREATE INDEX idx_detections_camera ON person_detections(camera_id, timestamp);
  CREATE INDEX idx_detections_timestamp ON person_detections(timestamp);
  CREATE INDEX idx_detections_cluster ON person_detections(cluster_id);

  CREATE TABLE IF NOT EXISTS unknown_clusters (
      cluster_id     INTEGER PRIMARY KEY AUTOINCREMENT,
      representative_encoding BLOB,        -- centroid encoding
      suggested_name TEXT,                  -- system suggestion (nullable)
      confirmed      INTEGER DEFAULT 0,    -- 0=pending, 1=confirmed
      created_at     TEXT NOT NULL DEFAULT (datetime('now'))
  );
  ```

- New file: `infrastructure/database/person_log_db.py`
- `PersonLogDB` class:
  - `__init__(db_path)` — open SQLite in WAL mode, run schema migration.
  - `insert_detection(entry: PersonLogEntry)`
  - `get_detections(person_name=None, camera_id=None, start=None, end=None, limit=100, offset=0)`
  - `get_person_names() -> list[str]`
  - `get_detection_by_id(detection_id)`
  - `update_person_name(detection_id, name)`
  - `delete_detection(detection_id)`
  - `delete_before(cutoff_date) -> int` — return count deleted.
  - `get_unidentified_with_encodings() -> list` — for clustering.

### 2.5 Redis Streams consumer

- New file: `infrastructure/messaging/redis_stream_consumer.py`
- `RedisStreamConsumer` class:
  - Runs as a dedicated process/thread.
  - `XREAD` from Redis stream in a loop (blocking read).
  - Deserialize `PersonLogEntry`, insert into `PersonLogDB`.
  - Single consumer = single writer to SQLite (no contention).
  - Acknowledge processed messages.

### 2.6 Consumer entry point

- New file: `interfaces/run_consumer.py`
  - CLI entry point that starts the Redis→SQLite consumer.
  - Reads config from `config.yaml`.

**Files created/modified:**
| File | Action |
|------|--------|
| `domain/detection/events.py` | Modify (add PersonLogEntry) |
| `infrastructure/storage/person_image_storage.py` | Create |
| `infrastructure/messaging/redis_stream_producer.py` | Create |
| `infrastructure/database/schema.sql` | Create |
| `infrastructure/database/person_log_db.py` | Create |
| `infrastructure/messaging/redis_stream_consumer.py` | Create |
| `interfaces/run_consumer.py` | Create |
| `application/camera_worker.py` | Modify (publish to Redis after detection) |

---

## Phase 3: Multi-Camera Orchestration

**Goal:** Run detection on up to 12 camera streams simultaneously.

### 3.1 Multi-camera orchestrator

- New file: `application/multi_camera_orchestrator.py`
- `MultiCameraOrchestrator` class:
  - Reads camera list from config.
  - Spawns one `CameraWorker` per camera, each in its own thread or process.
  - Each worker has its own `DetectionPipeline` + `IdentitySmoother`.
  - All workers share a single `RedisStreamProducer` (thread-safe).
  - Manages lifecycle: start, stop, health checks, restart on failure.
  - Logs per-camera FPS and detection stats.

### 3.2 Multi-camera CLI entry point

- New file: `interfaces/watch_all_streams.py`
  - Starts the `MultiCameraOrchestrator` from config.
  - Graceful shutdown on SIGINT/SIGTERM.

### 3.3 Update single-camera CLI

- `interfaces/watch_stream.py` remains for single-camera use (debugging, testing).
- Add `--config` flag to both CLIs.

**Files created/modified:**
| File | Action |
|------|--------|
| `application/multi_camera_orchestrator.py` | Create |
| `interfaces/watch_all_streams.py` | Create |
| `interfaces/watch_stream.py` | Modify (add --config flag) |

---

## Phase 4: Unknown Person Clustering

**Goal:** Group unidentified people by face similarity; suggest identities for user confirmation.

### 4.1 Clustering logic

- New file: `application/unknown_person_clusterer.py`
- `UnknownPersonClusterer` class:
  - `cluster(db: PersonLogDB, distance_threshold: float)`:
    - Fetch all unidentified detections with face encodings from SQLite.
    - Compute pairwise L2 distances.
    - Agglomerative clustering (or DBSCAN) with configurable distance threshold.
    - Assign `cluster_id` to each detection row.
    - Compute centroid encoding per cluster, store in `unknown_clusters` table.
  - `suggest_identities(db: PersonLogDB, encodings_path: str, threshold: float)`:
    - Compare cluster centroids against known encodings from `encodings.pkl`.
    - If distance < threshold, set `suggested_name` on the cluster.
  - Runs periodically (e.g., every hour or triggered after N new detections).

### 4.2 Confirmation workflow

- Handled in the API/UI layer (Phase 5).
- When user confirms a suggestion:
  - Update all detections in the cluster with `person_name`.
  - Mark cluster as `confirmed = 1`.

**Files created/modified:**
| File | Action |
|------|--------|
| `application/unknown_person_clusterer.py` | Create |
| `infrastructure/database/person_log_db.py` | Modify (add cluster query/update methods) |

---

## Phase 5: GraphQL API

**Goal:** Expose person detection data via a GraphQL API for the frontend.

### 5.1 Technology choice

- **Strawberry GraphQL** + **FastAPI** (async, lightweight, good typing support).
- New dependencies: `strawberry-graphql[fastapi]`, `fastapi`, `uvicorn`.

### 5.2 GraphQL schema

- New file: `interfaces/api/schema.py`
  ```graphql
  type Person {
    name: String!
    detectionCount: Int!
    lastSeen: DateTime
    firstSeen: DateTime
  }

  type Detection {
    detectionId: String!
    timestamp: DateTime!
    cameraId: Int!
    cameraLabel: String!
    personName: String
    confidence: Float
    faceCropUrl: String
    bodyCropUrl: String
    clusterId: Int
  }

  type UnknownCluster {
    clusterId: Int!
    detectionCount: Int!
    suggestedName: String
    confirmed: Boolean!
    representativeImageUrl: String
  }

  type Query {
    persons(limit: Int, offset: Int): [Person!]!
    detections(
      personName: String
      cameraId: Int
      startDate: DateTime
      endDate: DateTime
      limit: Int
      offset: Int
    ): [Detection!]!
    detection(detectionId: String!): Detection
    unknownClusters: [UnknownCluster!]!
    clusterDetections(clusterId: Int!): [Detection!]!
    cameras: [Camera!]!
    recentActivity(limit: Int): [Detection!]!
  }

  type Mutation {
    updatePersonName(detectionId: String!, name: String!): Detection!
    confirmCluster(clusterId: Int!, personName: String!): UnknownCluster!
    rejectClusterSuggestion(clusterId: Int!): UnknownCluster!
    deleteDetection(detectionId: String!): Boolean!
  }
  ```

### 5.3 API server

- New file: `interfaces/api/server.py` — FastAPI app with Strawberry GraphQL mount.
- New file: `interfaces/api/resolvers.py` — Query/mutation resolver implementations using `PersonLogDB`.
- Static file serving for face/body crop images.

### 5.4 API entry point

- New file: `interfaces/run_api.py` — CLI to start the API server via uvicorn.

**Files created/modified:**
| File | Action |
|------|--------|
| `interfaces/api/__init__.py` | Create |
| `interfaces/api/schema.py` | Create |
| `interfaces/api/resolvers.py` | Create |
| `interfaces/api/server.py` | Create |
| `interfaces/run_api.py` | Create |
| `requirements.txt` | Modify (add strawberry-graphql, fastapi, uvicorn) |

---

## Phase 6: Frontend UI

**Goal:** Web UI for browsing, filtering, and managing person detections.

### 6.1 Technology

- React + Apollo Client (GraphQL) + Vite (build tool).
- Separate `frontend/` directory at project root.

### 6.2 Pages / Views

| Page | Description |
|------|-------------|
| **Dashboard** (default) | Recent activity feed. Cards showing face crop, name/unknown, camera, timestamp. Auto-refreshes. |
| **Person Detail** | All appearances of a specific person. Timeline view with date/time filters. |
| **Unknown Clusters** | List of unknown clusters with representative images. "Suggest" badges for system-identified matches. Confirm/reject buttons. |
| **Search / Filter** | Filter detections by person, camera, date range. Sortable table view. |
| **Detection Detail** | Full detail for a single detection: face crop, body crop, metadata. Edit name, delete. |

### 6.3 Key components

- `DetectionCard` — thumbnail + name + camera + time.
- `PersonList` — sidebar or dropdown for filtering by person.
- `DateRangeFilter` — date/time picker for filtering.
- `ClusterReviewPanel` — side-by-side comparison of cluster members with confirm/reject actions.
- `ImageViewer` — lightbox for face/body crops.

**Files created:**
| File | Action |
|------|--------|
| `frontend/` | Create (React app scaffolded via Vite) |
| `frontend/src/pages/Dashboard.tsx` | Create |
| `frontend/src/pages/PersonDetail.tsx` | Create |
| `frontend/src/pages/UnknownClusters.tsx` | Create |
| `frontend/src/pages/DetectionDetail.tsx` | Create |
| `frontend/src/components/` | Create (shared components) |

---

## Phase 7: Data Retention & Cleanup

**Goal:** Automatically purge old records and associated images.

### 7.1 Retention manager

- New file: `application/retention_manager.py`
- `RetentionManager` class:
  - `cleanup(db: PersonLogDB, image_storage: PersonImageStorage, retention_days: int)`:
    - Query detections older than cutoff.
    - Delete associated face/body images from filesystem.
    - Delete rows from `person_detections`.
    - Delete orphaned clusters from `unknown_clusters`.
    - Log how many records/images were purged.

### 7.2 Scheduled execution

- Run as part of the consumer process (using `threading.Timer` or `schedule` library).
- Configurable interval from `config.yaml` (`retention.cleanup_interval_hours`).
- Alternatively, run as a standalone CLI: `interfaces/run_cleanup.py`.

**Files created/modified:**
| File | Action |
|------|--------|
| `application/retention_manager.py` | Create |
| `interfaces/run_cleanup.py` | Create (optional standalone CLI) |
| `infrastructure/messaging/redis_stream_consumer.py` | Modify (schedule periodic cleanup) |

---

## Phase 8: Dockerization

**Goal:** All components running in Docker Compose.

### 8.1 Docker images

| Service | Base Image | Dockerfile |
|---------|-----------|------------|
| `detector` | python:3.12-slim + torch/ultralytics | `Dockerfile.detector` |
| `consumer` | python:3.12-slim | `Dockerfile.consumer` |
| `api` | python:3.12-slim + fastapi | `Dockerfile.api` |
| `frontend` | node:20 (build) → nginx (serve) | `frontend/Dockerfile` |
| `redis` | redis:7-alpine | (official image) |

### 8.2 Docker Compose

- New file: `docker-compose.yml`
  ```yaml
  services:
    redis:
      image: redis:7-alpine
      ports: ["6379:6379"]
      volumes: [redis-data:/data]

    detector:
      build: { dockerfile: Dockerfile.detector }
      depends_on: [redis]
      volumes:
        - ./config.yaml:/app/config.yaml:ro
        - ./faces-output:/app/faces-output:ro
        - ./recordings:/app/recordings
      env_file: .env
      deploy:
        resources:
          limits: { memory: 4G }

    consumer:
      build: { dockerfile: Dockerfile.consumer }
      depends_on: [redis]
      volumes:
        - ./config.yaml:/app/config.yaml:ro
        - ./recordings:/app/recordings
        - surveillance-db:/app/data
      env_file: .env

    api:
      build: { dockerfile: Dockerfile.api }
      depends_on: [consumer]
      ports: ["8000:8000"]
      volumes:
        - ./config.yaml:/app/config.yaml:ro
        - ./recordings:/app/recordings:ro
        - surveillance-db:/app/data:ro

    frontend:
      build: { context: ./frontend }
      ports: ["3000:80"]
      depends_on: [api]

  volumes:
    redis-data:
    surveillance-db:
  ```

### 8.3 Fold existing face export Docker

- Keep `docker-compose.faces.yml` separate (one-off tool, different lifecycle).
- Reference it in documentation as a prerequisite step (generate `encodings.pkl` before running detection).

**Files created/modified:**
| File | Action |
|------|--------|
| `Dockerfile.detector` | Create |
| `Dockerfile.consumer` | Create |
| `Dockerfile.api` | Create |
| `frontend/Dockerfile` | Create |
| `docker-compose.yml` | Create |

---

## Dependency Graph

```
Phase 0 (Config & Refactor)
   │
   ├──→ Phase 1 (Smoothing)
   │        │
   │        ▼
   │    Phase 2 (Redis + SQLite)
   │        │
   │        ├──→ Phase 3 (Multi-Camera)
   │        │
   │        ├──→ Phase 4 (Clustering)
   │        │        │
   │        │        ▼
   │        ├──→ Phase 5 (GraphQL API)
   │        │        │
   │        │        ▼
   │        │    Phase 6 (Frontend UI)
   │        │
   │        └──→ Phase 7 (Retention)
   │
   └──→ Phase 8 (Docker) — can start after Phase 2, finalized after Phase 6
```

Phases 3, 4, 5, and 7 can be developed in parallel once Phase 2 is complete.

---

## New Dependencies

| Package | Phase | Purpose |
|---------|-------|---------|
| `pyyaml` | 0 | Config file parsing |
| `redis` | 2 | Redis Streams client |
| `strawberry-graphql[fastapi]` | 5 | GraphQL API |
| `fastapi` | 5 | API framework |
| `uvicorn` | 5 | ASGI server |
| `scikit-learn` | 4 | DBSCAN/agglomerative clustering |
| `schedule` | 7 | Periodic cleanup scheduling (optional) |

## Summary of All New Files

```
config.yaml
infrastructure/config.py
application/camera_worker.py
application/detection_smoother.py
application/multi_camera_orchestrator.py
application/unknown_person_clusterer.py
application/retention_manager.py
infrastructure/storage/person_image_storage.py
infrastructure/messaging/redis_stream_producer.py
infrastructure/messaging/redis_stream_consumer.py
infrastructure/database/schema.sql
infrastructure/database/person_log_db.py
interfaces/watch_all_streams.py
interfaces/run_consumer.py
interfaces/run_api.py
interfaces/run_cleanup.py
interfaces/api/__init__.py
interfaces/api/schema.py
interfaces/api/resolvers.py
interfaces/api/server.py
frontend/                          (React app)
Dockerfile.detector
Dockerfile.consumer
Dockerfile.api
docker-compose.yml
```
