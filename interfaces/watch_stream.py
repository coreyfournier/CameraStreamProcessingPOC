"""
Camera Stream Person Detection Script
Connects to a camera source (Synology Surveillance Station or ONVIF), reads
the stream, detects people, optionally matches faces against a known-faces
database, and records annotated video clips.
"""

import argparse
import os
import time

import cv2
from dotenv import load_dotenv

from application.detection_pipeline import DetectionPipeline
from application.stream_processor import draw_detections
from domain.detection.events import FaceMatchResult, PersonDetection
from infrastructure.camera.opencv_frame_reader import OpenCVFrameReader
from infrastructure.camera.synology_camera_source import SynologyCameraSource
from infrastructure.camera.onvif_camera_source import OnvifCameraSource
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

ONVIF_CONFIG = {
    "ip": os.environ.get('ONVIF_IP'),
    "port": os.environ.get('ONVIF_PORT', '80'),
    "username": os.environ.get('ONVIF_USERNAME'),
    "password": os.environ.get('ONVIF_PASSWORD'),
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
    parser.add_argument(
        "--source-type", choices=["synology", "onvif"], default="synology",
        help="Live camera source type (default: synology)",
    )
    parser.add_argument(
        "--onvif-profile", type=int, default=0,
        help="ONVIF media profile index to use (default: 0)",
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
        try:
            if args.source_type == 'synology':
                source = SynologyCameraSource(SYNOLOGY_CONFIG)
                cap = source.open(camera_id)
            else:
                source = OnvifCameraSource(ONVIF_CONFIG)
                cap = source.open(profile_index=args.onvif_profile)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            pipeline.stop()
            return

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
