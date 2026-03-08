"""ONVIF camera source.

Uses onvif-zeep to retrieve an RTSP stream URI from any ONVIF-compliant
camera and opens it with OpenCV/FFMPEG.
"""

import os
import re

import cv2


class OnvifCameraSource:
    def __init__(self, config: dict) -> None:
        """
        Args:
            config: dict with keys ip, port, username, password.
        """
        self.__config = config

    def get_rtsp_url(self, profile_index: int = 0) -> str | None:
        """Return the RTSP URL for the camera without opening a VideoCapture."""
        ip = self.__config["ip"]
        port = int(self.__config["port"])
        username = self.__config["username"]
        password = self.__config["password"]

        try:
            from onvif import ONVIFCamera
        except ImportError:
            return None

        cam = ONVIFCamera(ip, port, username, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            return None

        token = profiles[profile_index].token
        stream_setup = {
            "StreamSetup": {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            },
            "ProfileToken": token,
        }
        uri = media.GetStreamUri(stream_setup).Uri
        return self._inject_credentials(uri, username, password)

    def open(self, profile_index: int = 0) -> cv2.VideoCapture:
        """Connect to the ONVIF camera and return an opened VideoCapture.

        Args:
            profile_index: Index into the camera's media profile list (default 0).

        Returns:
            An opened cv2.VideoCapture reading the RTSP stream.

        Raises:
            RuntimeError: If the stream cannot be opened.
        """
        ip = self.__config["ip"]
        port = int(self.__config["port"])
        username = self.__config["username"]
        password = self.__config["password"]

        try:
            from onvif import ONVIFCamera
        except ImportError as exc:
            raise RuntimeError(
                "onvif-zeep is not installed. Run: pip install onvif-zeep"
            ) from exc

        print(f"Connecting to ONVIF camera at {ip}:{port}...")
        cam = ONVIFCamera(ip, port, username, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()

        if not profiles:
            raise RuntimeError("ONVIF camera returned no media profiles")

        token = profiles[profile_index].token
        stream_setup = {
            "StreamSetup": {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            },
            "ProfileToken": token,
        }
        uri = media.GetStreamUri(stream_setup).Uri
        print(f"ONVIF stream URI: {uri}")

        rtsp_url = self._inject_credentials(uri, username, password)

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        print("Opening ONVIF RTSP stream...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open ONVIF stream: {rtsp_url}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @staticmethod
    def _inject_credentials(uri: str, username: str, password: str) -> str:
        """Inject username:password into an RTSP URL if not already present."""
        if "@" in uri:
            return uri
        return re.sub(r"^(rtsp://)", rf"\1{username}:{password}@", uri)
