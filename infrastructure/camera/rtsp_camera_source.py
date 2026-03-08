"""Direct RTSP camera source.

Builds an RTSP URL from configuration and opens it with OpenCV/FFmpeg.
No ONVIF or Synology discovery required.
"""

import os
import re
from urllib.parse import quote

import cv2


class RtspCameraSource:
    def __init__(self, config: dict) -> None:
        """
        Args:
            config: dict with keys ip, port, username, password, path (optional).
        """
        self.__config = config

    def get_rtsp_url(self) -> str:
        """Build and return the RTSP URL from config."""
        ip = self.__config.get("ip", "")
        port = int(self.__config.get("port", 554))
        username = self.__config.get("username", "")
        password = self.__config.get("password", "")
        path = self.__config.get("path", "")

        # URL-encode username and password to handle special characters
        creds = ""
        if username:
            creds = f"{quote(username, safe='')}:{quote(password, safe='')}@"

        return f"rtsp://{creds}{ip}:{port}{path}"

    def open(self) -> cv2.VideoCapture:
        """Open and return a VideoCapture for the RTSP stream.

        Raises:
            RuntimeError: If the stream cannot be opened.
        """
        rtsp_url = self.get_rtsp_url()
        print(f"Opening RTSP stream: rtsp://***@{self.__config.get('ip')}:{self.__config.get('port', 554)}{self.__config.get('path', '')}")

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream at {self.__config.get('ip')}:{self.__config.get('port', 554)}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
