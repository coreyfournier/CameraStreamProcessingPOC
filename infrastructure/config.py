"""Centralized YAML configuration loader with environment variable interpolation.

Loads config.yaml, resolves ${ENV_VAR} references from os.environ (after
loading .env via python-dotenv), and provides helper functions to extract
typed sub-dicts for each subsystem.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in strings, lists, and dicts."""
    if isinstance(value, str):
        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")
        return _ENV_VAR_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load a YAML config file and resolve environment variable references.

    Calls ``load_dotenv()`` first so that ``.env`` values are available for
    interpolation.  Any ``${VAR_NAME}`` token in a string value is replaced
    with the corresponding environment variable (empty string if unset).

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        The fully-resolved configuration dictionary.
    """
    load_dotenv()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    return _resolve_env_vars(raw)


# ── Helper accessors ─────────────────────────────────────────────────


def get_camera_configs(config: dict) -> list[dict]:
    """Return the list of camera definitions."""
    return config.get("cameras", [])


def get_synology_config(config: dict) -> dict:
    """Return the synology section, mapped to SynologyCameraSource keys."""
    syn = config.get("synology", {})
    return {
        "ip_address": syn.get("ip", ""),
        "port": syn.get("port", "5001"),
        "username": syn.get("username", ""),
        "password": syn.get("password", ""),
        "secure": syn.get("secure", True),
        "cert_verify": syn.get("cert_verify", True),
        "dsm_version": syn.get("dsm_version", 7),
        "otp_code": None,
    }


def get_onvif_config(config: dict) -> dict:
    """Return the ONVIF section, mapped to OnvifCameraSource keys."""
    onvif = config.get("onvif", {})
    # Also check per-camera overrides — fall back to env-based defaults
    return {
        "ip": onvif.get("ip", os.environ.get("ONVIF_IP", "")),
        "port": onvif.get("port", os.environ.get("ONVIF_PORT", "80")),
        "username": onvif.get("username", os.environ.get("ONVIF_USERNAME", "")),
        "password": onvif.get("password", os.environ.get("ONVIF_PASSWORD", "")),
    }


def get_rtsp_config(config: dict) -> dict:
    """Return the RTSP section, mapped to RtspCameraSource keys."""
    rtsp = config.get("rtsp", {})
    return {
        "ip": rtsp.get("ip", os.environ.get("RTSP_IP", os.environ.get("ONVIF_IP", ""))),
        "port": rtsp.get("port", os.environ.get("RTSP_PORT", os.environ.get("ONVIF_PORT", "554"))),
        "username": rtsp.get("username", os.environ.get("RTSP_USERNAME", os.environ.get("ONVIF_USERNAME", ""))),
        "password": rtsp.get("password", os.environ.get("RTSP_PASSWORD", os.environ.get("ONVIF_PASSWORD", ""))),
        "path": rtsp.get("path", os.environ.get("RTSP_PATH", "")),
    }


def get_detection_config(config: dict) -> dict:
    """Return the detection section."""
    return config.get("detection", {
        "confidence_threshold": 0.5,
        "encodings_path": "./faces-output/encodings.pkl",
        "match_tolerance": 0.9,
        "match_skip_frames": 5,
        "match_min_confidence": 0.5,
    })


def get_recording_config(config: dict) -> dict:
    """Return the recording section."""
    return config.get("recording", {
        "enabled": True,
        "output_dir": "./recordings",
        "clip_duration": 30,
    })


def get_smoothing_config(config: dict) -> dict:
    """Return the smoothing section."""
    return config.get("smoothing", {
        "window_size": 10,
        "min_hit_ratio": 0.7,
        "min_avg_confidence": 0.7,
    })


def get_redis_config(config: dict) -> dict:
    """Return the redis section."""
    return config.get("redis", {
        "host": "localhost",
        "port": 6379,
        "stream_name": "person_detections",
    })


def get_database_config(config: dict) -> dict:
    """Return the database section."""
    return config.get("database", {
        "path": "./data/surveillance.db",
    })


def get_storage_config(config: dict) -> dict:
    """Return the storage section."""
    return config.get("storage", {
        "person_images_dir": "./recordings/persons",
    })


def get_retention_config(config: dict) -> dict:
    """Return the retention section."""
    return config.get("retention", {
        "days": 30,
        "cleanup_interval_hours": 24,
    })


def get_api_config(config: dict) -> dict:
    """Return the api section."""
    return config.get("api", {
        "host": "0.0.0.0",
        "port": 8000,
    })
