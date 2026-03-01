"""
Synology Surveillance Station Person Detection Script
Connects to Surveillance Station, reads camera stream, detects people,
optionally matches faces against a known-faces database, and records
annotated video clips.
"""

import argparse
import json
import os
import time

import cv2
from dotenv import load_dotenv

from application.detection_pipeline import DetectionPipeline
from application.stream_processor import draw_detections
from domain.detection.events import FaceMatchResult, PersonDetection
from infrastructure.camera.opencv_frame_reader import OpenCVFrameReader
from infrastructure.camera.synology_camera_source import CameraSource
from infrastructure.recording.avi_clip_writer import AviClipWriter

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
ENCODINGS_PATH = "./faces-output/encodings.pkl"  # Path to face encodings database
MATCH_SKIP_FRAMES = 5                           # Attempt face match every Nth person-detection frame
MATCH_TOLERANCE = 0.9                           # Max face distance for a match (lower = stricter)
MATCH_MIN_CONFIDENCE = 0.5               # Ignore person detections below this for matching


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synology Surveillance Station person detection with optional face matching."
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_ID,
        help=f"Camera ID in Surveillance Station (default: {CAMERA_ID})",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to a video file to use instead of a live camera stream",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop the source video file when it ends (only with --source)",
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

    # Open video source (file or live stream)
    is_file = args.source is not None
    if is_file:
        print(f"Opening video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {args.source}")
            pipeline.stop()
            return
    else:
        # Connect to Synology
        cs = CameraSource(SYNOLOGY_CONFIG)
        ss = cs.connect()

        # Get stream URL
        stream_url = cs.get_camera_stream_url(ss, camera_id)
        print(json.dumps(stream_url))
        print(f"Stream URL: {stream_url}")

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

    source_label = args.source if is_file else "stream"
    print(f"Opened {source_label}: {width}x{height} @ {fps}fps")

    # Background frame reader for live streams (not needed for files)
    reader = None
    if not is_file:
        reader = OpenCVFrameReader(cap).start()

    # AVI clip writer — auto-rotates every RECORD_DURATION seconds
    clip_writer = AviClipWriter(
        output_dir=OUTPUT_DIR,
        fps=fps,
        width=width,
        height=height,
        clip_duration_seconds=RECORD_DURATION,
    )

    print("Starting detection loop (press 'q' to quit)...")
    frame_count = 0

    try:
        while True:
            if reader:
                ret, frame = reader.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    if is_file and args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("End of video file.")
                    break

            frame_count += 1

            # Detect people (synchronous)
            detections = pipeline.process_frame(frame, frame_count)

            # Poll latest face-match results (non-blocking, from background thread)
            match_results = pipeline.get_latest_matches()

            # Draw detections on frame
            annotated_frame = draw_detections(frame.copy(), detections, match_results)

            # Write to video file
            clip_writer.write(annotated_frame)

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

            # Display frame (optional - comment out for headless)
            cv2.imshow("Surveillance Detection", annotated_frame)

            # Pace file playback to original fps; live streams use 1ms
            wait_ms = max(1, int(1000 / fps)) if is_file else 1
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Cleanup
        if reader:
            reader.stop()
        pipeline.stop()
        clip_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
