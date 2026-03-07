"""CLI entry point for the Redis-to-SQLite detection consumer.

Reads person detection messages from a Redis Stream and persists them
to the SQLite database.

Usage:
    python -m interfaces.run_consumer [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from infrastructure.config import (
    load_config,
    get_redis_config,
    get_database_config,
    get_storage_config,
    get_retention_config,
)
from infrastructure.database.person_log_db import PersonLogDB
from infrastructure.messaging.redis_stream_consumer import RedisStreamConsumer
from infrastructure.storage.person_image_storage import PersonImageStorage
from application.retention_manager import RetentionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Person detection Redis consumer")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    redis_cfg = get_redis_config(config)
    db_cfg = get_database_config(config)
    storage_cfg = get_storage_config(config)
    retention_cfg = get_retention_config(config)

    db = PersonLogDB(db_cfg["path"])
    image_storage = PersonImageStorage(storage_cfg["person_images_dir"])

    consumer = RedisStreamConsumer(
        host=redis_cfg["host"],
        port=int(redis_cfg["port"]),
        stream_name=redis_cfg["stream_name"],
        db=db,
    )

    retention_manager = RetentionManager(
        db=db,
        image_storage=image_storage,
        retention_days=int(retention_cfg["days"]),
    )

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(signum: int, frame: object) -> None:
        logger.info("Shutdown signal received")
        retention_manager.stop()
        consumer.stop()
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    consumer.start()
    retention_manager.start_scheduled(
        interval_hours=float(retention_cfg["cleanup_interval_hours"])
    )
    logger.info("Consumer and retention scheduler running. Press Ctrl+C to stop.")

    # Block the main thread until a signal arrives
    signal.pause() if hasattr(signal, "pause") else _wait_forever()


def _wait_forever() -> None:
    """Fallback for Windows where signal.pause() is unavailable."""
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
