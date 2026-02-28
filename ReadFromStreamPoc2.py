"""
Synology Surveillance Station Person Detection Script
Connects to Surveillance Station, reads camera stream, detects people,
optionally matches faces against a known-faces database, and records
annotated video clips.
"""

import argparse
import cv2
import threading
import time
from datetime import datetime
from CameraSource import CameraSource
import os
import json
from dotenv import load_dotenv

from detection_pipeline import DetectionPipeline
from events import FaceMatchResult, PersonDetection

load_dotenv()

# ============== CONFIGURATION ==============
SYNOLOGY_CONFIG = {
    "ip_address": os.environ.get('SYNOLOGY_IP'),       # e.g., "192.168.1.100"
    "port": os.environ.get('SYNOLOGY_PORT'),            # Default: 5000 (HTTP) or 5001 (HTTPS)
    "username": os.environ.get('SYNOLOGY_USERNAME'),
    "password": os.environ.get('SYNOLOGY_PASSWORD'),
    "secure": True,                   # Set True for HTTPS
    "cert_verify": True,
    "dsm_version": 7,                  # DSM version (6 or 7)
    "otp_code": None                   # 2FA code if enabled
}

CAMERA_ID = 14                          # Default camera ID in Surveillance Station
OUTPUT_DIR = "./recordings"            # Output directory for videos
DETECTION_CONFIDENCE = 0.5             # Minimum confidence threshold
RECORD_DURATION = 30                   # Seconds per video clip

# Face matching settings
ENCODINGS_PATH = "./faces/encodings.pkl"  # Path to face encodings database
MATCH_SKIP_FRAMES = 5                     # Attempt face match every Nth person-detection frame
MATCH_TOLERANCE = 0.6                     # Max face distance for a match (lower = stricter)
MATCH_MIN_CONFIDENCE = 0.5               # Ignore person detections below this for matching

# Colors (BGR)
COLOR_MATCHED = (0, 200, 0)     # Green — known face
COLOR_UNMATCHED = (0, 220, 255) # Yellow — person, unknown face


# ============== THREADED FRAME READER ==============

class FrameReader:
    """Read RTSP frames on a background thread so the main loop never
    blocks waiting on network I/O.  Only the most recent frame is kept,
    older frames are silently dropped.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._reader, name="frame-reader", daemon=True
        )
        self._thread.start()
        return self

    def read(self):
        """Return the most recent (ret, frame) — non-blocking."""
        with self._lock:
            return self._ret, self._frame

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame


# ============== DRAWING ==============

def _box_center(box):
    """Return (cx, cy) of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _box_distance_sq(box_a, box_b):
    """Squared Euclidean distance between two box centers."""
    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)
    return (ax - bx) ** 2 + (ay - by) ** 2


def _associate_matches(detections, match_results):
    """Associate async match results to current detections by box proximity.

    Returns a dict mapping detection index → FaceMatchResult (or None).
    """
    associations = {}
    if not match_results:
        return associations

    # For each current detection, find the closest match result by box center
    used = set()
    for i, det in enumerate(detections):
        best_j = None
        best_dist = float("inf")
        for j, mr in enumerate(match_results):
            if j in used:
                continue
            d = _box_distance_sq(det.box, mr.person_detection.box)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j is not None:
            associations[i] = match_results[best_j]
            used.add(best_j)

    return associations


def draw_detections(frame, detections, match_results):
    """Draw bounding boxes and labels on frame.

    - Green box + person name for matched faces
    - Yellow box + "Person" for unmatched detections
    """
    associations = _associate_matches(detections, match_results)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.box
        mr = associations.get(i)

        if mr and mr.matched:
            color = COLOR_MATCHED
            label = f"{mr.person_name} ({mr.confidence:.0%})"
        else:
            color = COLOR_UNMATCHED
            label = f"Person ({det.confidence:.0%})"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
        )

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame, timestamp, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synology Surveillance Station person detection with optional face matching."
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_ID,
        help=f"Camera ID in Surveillance Station (default: {CAMERA_ID})",
    )
    return parser.parse_args()


# ============== MAIN PROCESSING LOOP ==============
def main():
    args = parse_args()
    camera_id = args.camera

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialise detection pipeline
    print("Initialising detection pipeline...")
    pipeline = DetectionPipeline(
        confidence_threshold=DETECTION_CONFIDENCE,
        encodings_path=ENCODINGS_PATH,
        match_tolerance=MATCH_TOLERANCE,
        match_min_confidence=MATCH_MIN_CONFIDENCE,
        match_skip_frames=MATCH_SKIP_FRAMES,
    )
    pipeline.start()

    # Connect to Synology
    cs = CameraSource(SYNOLOGY_CONFIG)
    ss = cs.connect()

    # Get stream URL
    stream_url = cs.get_camera_stream_url(ss, camera_id)
    print(json.dumps(stream_url))
    print(f"Stream URL: {stream_url}")

    # Open video stream
    # Force RTSP over TCP to avoid UDP packet-loss artifacts
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    print("Opening video stream...")
    cap = None

    if 'rtspPath' in stream_url:
        print("trying RTSP over TCP (full resolution)...")
        cap = cv2.VideoCapture(stream_url['rtspPath'], cv2.CAP_FFMPEG)

    if not cap or not cap.isOpened():
        if 'rtspOverHttpPath' in stream_url:
            print("trying RTSP over HTTP...")
            cap = cv2.VideoCapture(stream_url['rtspOverHttpPath'], cv2.CAP_FFMPEG)

    if not cap or not cap.isOpened():
        print("Falling back to MJPEG...")
        cap = cv2.VideoCapture(stream_url['mjpegHttpPath'])

    if not cap.isOpened():
        print("Error: Could not open video stream")
        pipeline.stop()
        return

    # Minimize internal buffer so we always get the latest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    print(f"Stream opened: {width}x{height} @ {fps}fps")

    # Start background frame reader so network I/O never blocks the loop
    reader = FrameReader(cap).start()

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    clip_start_time = time.time()
    clip_number = 0

    def start_new_clip():
        nonlocal out, clip_start_time, clip_number
        clip_number += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/detection_{timestamp}_{clip_number:04d}.mp4"
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        clip_start_time = time.time()
        print(f"Recording: {filename}")
        return filename

    current_file = start_new_clip()

    print("Starting detection loop (press 'q' to quit)...")
    frame_count = 0

    try:
        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Detect people (synchronous — MobileNet internally resizes to 300x300)
            detections = pipeline.process_frame(frame, frame_count)

            # Poll latest face-match results (non-blocking, from background thread)
            match_results = pipeline.get_latest_matches()

            # Draw detections on frame
            annotated_frame = draw_detections(frame.copy(), detections, match_results)

            # Write to video file
            out.write(annotated_frame)

            # Log detections
            if detections:
                names = []
                for mr in match_results:
                    if mr.matched:
                        names.append(mr.person_name)
                summary = f"{len(detections)} person(s)"
                if names:
                    summary += f" [{', '.join(names)}]"
                print(f"Frame {frame_count}: {summary}")

            # Start new clip if duration exceeded
            if time.time() - clip_start_time >= RECORD_DURATION:
                out.release()
                print(f"Saved clip: {current_file} ({frame_count} frames)")
                frame_count = 0
                current_file = start_new_clip()

            # Display frame (optional - comment out for headless)
            cv2.imshow("Surveillance Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Cleanup
        reader.stop()
        pipeline.stop()
        if out:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
