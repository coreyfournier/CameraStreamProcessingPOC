"""CLI entry point for manual retention cleanup.

Deletes person detection records and images older than the configured
retention period (or the --days override).

Usage:
    python -m interfaces.run_cleanup [--config config.yaml] [--days 30]
"""

from __future__ import annotations

import argparse
import logging

from infrastructure.config import (
    load_config,
    get_database_config,
    get_storage_config,
    get_retention_config,
)
from infrastructure.database.person_log_db import PersonLogDB
from infrastructure.storage.person_image_storage import PersonImageStorage
from application.retention_manager import RetentionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual retention cleanup")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Override retention days from config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    db_cfg = get_database_config(config)
    storage_cfg = get_storage_config(config)
    retention_cfg = get_retention_config(config)

    retention_days = args.days if args.days is not None else int(retention_cfg["days"])

    db = PersonLogDB(db_cfg["path"])
    image_storage = PersonImageStorage(storage_cfg["person_images_dir"])

    try:
        manager = RetentionManager(db, image_storage, retention_days=retention_days)
        result = manager.cleanup()
        print(
            f"Cleanup complete: {result['deleted_count']} record(s) deleted "
            f"(cutoff: {result['cutoff_date']})"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
