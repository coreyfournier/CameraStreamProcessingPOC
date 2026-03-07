"""CLI entry point for multi-camera monitoring.

Loads a YAML configuration file, creates a MultiCameraOrchestrator, starts
all configured cameras, and blocks until the user presses Ctrl+C.

Usage::

    python -m interfaces.watch_all_streams --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor multiple camera streams with person detection.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    from infrastructure.config import load_config
    from application.multi_camera_orchestrator import MultiCameraOrchestrator

    logger.info("Loading configuration from %s", args.config)
    config = load_config(args.config)

    orchestrator = MultiCameraOrchestrator(config)

    # Use an event so the main thread can be woken by SIGINT/SIGTERM
    shutdown_event = threading.Event()

    def _signal_handler(signum, frame):
        logger.info("Received signal %s, shutting down...", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    orchestrator.start()
    logger.info("All cameras started. Press Ctrl+C to stop.")

    # Block until shutdown is requested
    shutdown_event.wait()

    orchestrator.stop()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
