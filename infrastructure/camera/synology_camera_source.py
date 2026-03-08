"""Synology Surveillance Station camera source.

Handles connection, camera enumeration, stream URL retrieval, and URL
fixup (the NAS sometimes returns incorrect protocol/port in stream URLs).
"""

import cv2
import numpy as np
import time
import json
import os
import re
from datetime import datetime
from synology_api import surveillancestation


class SynologyCameraSource:
    def __init__(self, config: dict):
        self.__config = config

    def get_rtsp_url(self, camera_id: int) -> str | None:
        """Return the best RTSP URL for the camera, or None if unavailable."""
        ss = self.connect()
        stream_url = self.get_camera_stream_url(ss, camera_id)
        if 'rtspPath' in stream_url:
            return stream_url['rtspPath']
        if 'rtspOverHttpPath' in stream_url:
            return stream_url['rtspOverHttpPath']
        return None

    def open(self, camera_id: int) -> cv2.VideoCapture:
        """Connect to Synology and open a VideoCapture for the given camera.

        Tries RTSP over TCP → RTSP over HTTP → MJPEG in that order.
        Raises RuntimeError if no stream can be opened.
        """
        ss = self.connect()
        stream_url = self.get_camera_stream_url(ss, camera_id)
        print(json.dumps(stream_url))

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

        if not cap or not cap.isOpened():
            raise RuntimeError("Could not open any Synology camera stream")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def connect(self):
        """Establish connection to Synology Surveillance Station."""
        print("Connecting to Synology Surveillance Station...")

        ss = surveillancestation.SurveillanceStation(
            ip_address=self.__config["ip_address"],
            port=self.__config["port"],
            username=self.__config["username"],
            password=self.__config["password"],
            secure=self.__config["secure"],
            cert_verify=self.__config["cert_verify"],
            dsm_version=self.__config["dsm_version"],
            otp_code=self.__config["otp_code"],
            debug=True
        )

        cameras = ss.camera_list()['data']['cameras']
        for camera in cameras:
            print(f'{camera["id"]} {camera["ip"]} {camera["newName"]}\n')
            obj = self.getPath(ss, camera['id'])
            print(json.dumps(obj))
            print()

        print("Connected successfully!")
        return ss

    def get_camera_stream_url(self, ss, camera_id) -> any:
        """Get the RTSP or MJPEG stream URL for a camera."""
        # Get camera info
        camera_info = ss.get_camera_info(camera_id)
        camera = camera_info['data']['cameras'][0]

        # Try to get live view path
        snap_shot = ss.get_snapshot(camera_id)  # Gets snapshot URL pattern
        outputDir = os.path.join('.', 'camera')

        os.makedirs(outputDir, exist_ok=True)

        with open(os.path.join(outputDir, f'{camera["name"]}.jpg'), 'wb') as file:
            file.write(snap_shot)

        obj = self.getPath(ss, camera_id)
        print(json.dumps(obj))
        return obj

    def getPath(self, ss: surveillancestation.SurveillanceStation, cameraId: int):
        camera_object = None
        for camera in ss.get_live_path(cameraId)['data']:
            # Replaces the ip address with the dns entry
            for key in camera.keys():
                if key.endswith("Path"):
                    camera[key] = self.fixAddress(self.__config["ip_address"], camera[key])
            camera_object = camera

        return camera_object

    def fixAddress(self, dns: str, url: str):
        """The server doesn't use the defined security configuration and must be modified to support it.

        Args:
            dns: server dns to replace the ip address with
            url: url to modify

        Returns:
            Modified URL with corrected address and protocol.
        """
        temp = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', dns, url)
        temp = temp.replace(':5000', f":{self.__config['port']}")
        if self.__config['secure']:
            temp = temp.replace('http://', "https://")
        return temp
