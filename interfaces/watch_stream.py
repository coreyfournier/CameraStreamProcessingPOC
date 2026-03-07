"""
Camera Stream Person Detection Script
Connects to a camera source (Synology Surveillance Station or ONVIF), reads
the stream, detects people, optionally matches faces against a known-faces
database, and records annotated video clips.

Supports two modes:
  1. Config-file mode (--config config.yaml) — reads all settings from YAML.
  2. Legacy mode (no --config) — uses env vars and hardcoded defaults as before.
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

# ============== LEGACY DEFAULTS (used when no --config is provided) ==============
SYNOLOGY_CONFIG = {
    "ip_address": os.environ.get('SYNOLOGY_IP'),
    "port": os.environ.get('SYNOLOGY_PORT'),
    "username": os.environ.get('SYNOLOGY_USERNAME'),
    "password": os.environ.get('SYNOLOGY_PASSWORD'),
    "secure": True,
    "cert_verify": True,
    "dsm_version": 7,
    "otp_code": None,
}

ONVIF_CONFIG = {
    "ip": os.environ.get('ONVIF_IP'),
    "port": os.environ.get('ONVIF_PORT', '80'),
    "username": os.environ.get('ONVIF_USERNAME'),
    "password": os.environ.get('ONVIF_PASSWORD'),
}

CAMERA_ID = 14
OUTPUT_DIR = "./recordings"
DETECTION_CONFIDENCE = 0.5
RECORD_DURATION = 30

ENCODINGS_PATH = "./faces-output/encodings.pkl"
MATCH_SKIP_FRAMES = 5
MATCH_TOLERANCE = 0.9
MATCH_MIN_CONFIDENCE = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera stream person detection with optional face matching."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: None, uses env vars)",
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
    parser.add_argument(
        "--no-record", action="store_true",
        help="Disable saving annotated video clips to disk",
    )
    return parser.parse_args()


# ============== MAIN ==============
def main():
    args = parse_args()

    from application.camera_worker import CameraWorker

    if args.config:
        # ── Config-file mode ─────────────────────────────────────────
        from infrastructure.config import (
            load_config,
            get_camera_configs,
            get_detection_config,
            get_recording_config,
            get_synology_config,
            get_onvif_config,
            get_redis_config,
            get_storage_config,
            get_smoothing_config,
        )
        from infrastructure.messaging.redis_stream_producer import RedisStreamProducer
        from infrastructure.storage.person_image_storage import PersonImageStorage
        from application.detection_smoother import IdentitySmoother
        from domain.detection.events import PersonLogEntry
        import uuid
        from datetime import datetime

        config = load_config(args.config)
        detection_cfg = get_detection_config(config)
        recording_cfg = get_recording_config(config)
        synology_cfg = get_synology_config(config)
        onvif_cfg = get_onvif_config(config)
        redis_cfg = get_redis_config(config)
        storage_cfg = get_storage_config(config)
        smoothing_cfg = get_smoothing_config(config)

        # CLI overrides
        if args.no_record:
            recording_cfg["enabled"] = False

        cameras = get_camera_configs(config)
        if not cameras:
            cameras = [{
                "id": args.camera,
                "label": f"camera_{args.camera}",
                "source_type": args.source_type,
            }]

        cam_cfg = cameras[0]

        if args.camera != CAMERA_ID:
            cam_cfg["id"] = args.camera
        if args.source_type != "synology":
            cam_cfg["source_type"] = args.source_type

        # Set up Redis publishing + image storage
        producer = RedisStreamProducer(
            host=redis_cfg.get("host", "localhost"),
            port=int(redis_cfg.get("port", 6379)),
            stream_name=redis_cfg.get("stream_name", "person_detections"),
        )
        image_storage = PersonImageStorage(
            base_dir=storage_cfg.get("person_images_dir", "./recordings/persons"),
        )
        smoother = IdentitySmoother(
            window_size=smoothing_cfg.get("window_size", 10),
            min_hit_ratio=smoothing_cfg.get("min_hit_ratio", 0.7),
            min_avg_confidence=smoothing_cfg.get("min_avg_confidence", 0.7),
        )

        camera_label = cam_cfg.get("label", f"camera_{cam_cfg.get('id', 0)}")
        camera_id = cam_cfg.get("id", 0)

        def on_detection(label, detections, match_results):
            smoothed = smoother.smooth(detections, match_results)
            now_iso = datetime.now().isoformat()
            for si in smoothed:
                det_id = str(uuid.uuid4())
                face_path, body_path = image_storage.save(
                    detection_id=det_id,
                    face_crop=None,
                    body_crop=si.person_detection.person_crop,
                    timestamp=now_iso,
                )
                entry = PersonLogEntry(
                    detection_id=det_id,
                    timestamp=now_iso,
                    camera_id=camera_id,
                    camera_label=camera_label,
                    person_name=si.person_name if si.person_name != "Unknown" else None,
                    confidence=si.confidence,
                    face_crop_path=face_path,
                    body_crop_path=body_path,
                    face_encoding=None,
                    track_id=si.track_id,
                )
                producer.publish(entry)

        # Use --source from CLI, or fall back to source in camera config
        source_file = args.source or cam_cfg.get("source")
        loop_file = args.loop or cam_cfg.get("loop", False)

        worker = CameraWorker(
            camera_config=cam_cfg,
            detection_config=detection_cfg,
            recording_config=recording_cfg,
            synology_config=synology_cfg,
            onvif_config=onvif_cfg,
            source_file=source_file,
            loop_file=loop_file,
            onvif_profile=args.onvif_profile,
            show_display=True,
            on_detection=on_detection,
        )
        try:
            worker.run_blocking()
        finally:
            producer.close()

    else:
        # ── Legacy mode (no config file) ─────────────────────────────
        camera_config = {
            "id": args.camera,
            "label": f"camera_{args.camera}",
            "source_type": args.source_type,
        }
        detection_config = {
            "confidence_threshold": DETECTION_CONFIDENCE,
            "encodings_path": ENCODINGS_PATH,
            "match_tolerance": MATCH_TOLERANCE,
            "match_min_confidence": MATCH_MIN_CONFIDENCE,
            "match_skip_frames": MATCH_SKIP_FRAMES,
        }
        recording_config = {
            "enabled": not args.no_record,
            "output_dir": OUTPUT_DIR,
            "clip_duration": RECORD_DURATION,
        }

        worker = CameraWorker(
            camera_config=camera_config,
            detection_config=detection_config,
            recording_config=recording_config,
            synology_config=SYNOLOGY_CONFIG,
            onvif_config=ONVIF_CONFIG,
            source_file=args.source,
            loop_file=args.loop,
            onvif_profile=args.onvif_profile,
            show_display=True,
        )
        worker.run_blocking()


if __name__ == "__main__":
    main()
