# Surveillance Nation

I want to create an application / process that allows me to see who and what came to my house and when.

## Tasks

1. **Refactor stream processing into its own module** — Move the existing code that takes a stream in and does person identification into its own folder/module so it can be invoked independently from the main entry point.
2. **Dockerize all components** — All project components should be in Docker containers and orchestrated via Docker Compose.

## Features

### 1. Person recognition accuracy

1. Person identification is inaccurate and flip-flops back and forth. Create a smoothing algorithm that requires both:
   - A minimum **average confidence score** over a sliding window of frames.
   - A minimum **identification hit ratio** (e.g., identified in 7 of the last 10 frames) before locking in an identity.
   - Both thresholds should be configurable.

### 2. Person logging

1. A **person** can refer to an identified or unidentified individual. Identified people are those matched against the `encodings.pkl` file. Unidentified people are detected but not matched.
2. Create a stream processing pipeline using **Redis Streams** as the message broker to log person detections.
   - The stream should support handling a high volume of messages and processing them when time permits.
   - **Backpressure policy:** Keep all messages — buffer everything in Redis and process eventually. No messages are dropped.
3. Use **SQLite** (WAL mode) as the database to store person logs.
   - A single Redis Streams consumer writes to SQLite, avoiding concurrent-writer contention.
   - Architecture: `12 cameras -> Redis Streams (buffer) -> single consumer -> SQLite`
4. Each person detection record should store:
   - Timestamp of detection.
   - **Camera source** (camera ID + human-friendly location label, e.g., "front door", "driveway").
   - If the person is identified, link to their identity.
5. The person's **face crop** and **identified body crop** should be stored as two images on the **filesystem** (not in the database). The database stores file paths.
   - Image path convention: `recordings/persons/YYYY/MM/DD/{detection_id}_face.jpg`, `..._body.jpg`
6. The database should support quickly selecting a person and finding all timestamps when they were seen/appeared.
7. Unidentified people should be indexed and the system should attempt to cluster/link them to others in the system, but only with a high confidence (configurable). When a cluster is later identified:
   - The system **suggests** matches to the user.
   - A user must **manually confirm** via the UI before past rows are linked to the identity.
8. **Data retention:** A periodic cleanup job deletes records and associated images older than a configurable retention period (e.g., 30 days). Runs via a background task or scheduled job.

### 3. Person lookup

1. Create a front-end and back-end that supports reviewing persons and logs.
2. **Backend API:** GraphQL, allowing clients to select exactly which fields are returned.
3. **UI** (full CRUD — view, label/name unknowns, edit, delete):
   - View who appeared and when.
   - Filter by date, time, camera/source, and person.
   - Defaults to who was recently seen.
   - Allow naming/labeling unknown people.
   - Review and confirm/reject suggested identity matches for unknown clusters.

## Scale Requirements

- Design for at least **12 simultaneous camera streams** running person detection.
- SQLite single-writer is mitigated by Redis Streams buffering (single consumer pattern).
- If SQLite contention becomes an issue at scale, evaluate switching to PostgreSQL.
