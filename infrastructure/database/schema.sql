-- Person detection log schema

CREATE TABLE IF NOT EXISTS person_detections (
    detection_id   TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,              -- ISO 8601
    camera_id      INTEGER NOT NULL,
    camera_label   TEXT NOT NULL,
    person_name    TEXT,                       -- NULL if unidentified
    confidence     REAL NOT NULL,
    face_crop_path TEXT,
    body_crop_path TEXT,
    face_encoding  BLOB,                      -- serialized 512-d float32
    cluster_id     INTEGER REFERENCES unknown_clusters(cluster_id),
    track_id       INTEGER,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS unknown_clusters (
    cluster_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    representative_encoding BLOB NOT NULL,
    suggested_name          TEXT,
    confirmed               INTEGER NOT NULL DEFAULT 0,
    detection_count         INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_detections_person_ts
    ON person_detections (person_name, timestamp);

CREATE INDEX IF NOT EXISTS idx_detections_camera_ts
    ON person_detections (camera_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_detections_ts
    ON person_detections (timestamp);

CREATE INDEX IF NOT EXISTS idx_detections_cluster
    ON person_detections (cluster_id);
