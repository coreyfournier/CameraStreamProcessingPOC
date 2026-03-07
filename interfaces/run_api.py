"""CLI entry point for the GraphQL API server."""

from __future__ import annotations

import argparse

import uvicorn

from infrastructure.config import load_config, get_api_config
from interfaces.api.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Camera Surveillance GraphQL API")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    api_config = get_api_config(config)

    app = create_app(config)

    uvicorn.run(
        app,
        host=api_config.get("host", "0.0.0.0"),
        port=int(api_config.get("port", 8000)),
    )


if __name__ == "__main__":
    main()
