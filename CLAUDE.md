# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python proof-of-concept for real-time camera stream processing with object detection. Connects to a Synology Surveillance Station NAS via its API, reads RTSP/MJPEG camera streams, runs MobileNet-SSD object detection on each frame, and records annotated video clips.

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

**CameraSource.py** — Synology Surveillance Station client wrapper. Handles connection, camera enumeration, stream URL retrieval, and URL fixup (the NAS sometimes returns incorrect protocol/port in stream URLs, so `fixAddress()` corrects them).

**ReadFromStreamPoc2.py** — Main entry point. Loads MobileNet-SSD model, connects to Synology via `CameraSource`, reads frames in a loop, runs object detection, draws bounding boxes/labels, and writes annotated output to MP4 clips (30-second segments) in `./recordings/`.

**ReadFromStreamPoc1.py** — Simpler CLI alternative that takes an RTSP URL directly (no Synology integration). Useful for testing detection against any RTSP source.

**ExportLightroomFaces.py** — Standalone CLI tool that extracts named faces from an Adobe Lightroom Classic catalog (`.lrcat` SQLite database). Crops face regions from source photos (handling EXIF orientation), saves them organized by person name, and optionally computes 128-d face encodings via `face_recognition` for real-time matching.

```bash
# Export all faces (without encodings — no dlib needed)
python ExportLightroomFaces.py --catalog "path/to/My Catalog.lrcat" --output ./faces/ --skip-encodings

# Export a specific person with encodings
python ExportLightroomFaces.py --catalog "path/to/My Catalog.lrcat" --person "John Smith"

# Full options
python ExportLightroomFaces.py --catalog <path> [--output ./faces/] [--person "Name"] [--padding 40] [--min-size 50] [--skip-encodings]
```

Output structure: `faces/<Person_Name>/face_001.jpg`, `faces/encodings.pkl`, `faces/export_log.json`.

**MobileNetSSN/** — Pre-trained MobileNet-SSD Caffe model files (prototxt + caffemodel). Detects 21 object classes (person, car, dog, etc.) at 300x300 input resolution with a configurable confidence threshold (default 0.5).

## Environment Configuration

Synology credentials are loaded from `.env` (not committed):
- `ip_address` — NAS hostname/IP
- `port` — Surveillance Station port (5001 for HTTPS)
- `username` / `password` — NAS credentials

Hardcoded settings in ReadFromStreamPoc2.py: `CAMERA_ID`, `DETECTION_CONFIDENCE`, `RECORD_DURATION`, SSL options (`secure`, `cert_verify`, `dsm_version`).
