import cv2
import numpy as np
import time
from datetime import datetime
from synology_api import surveillancestation
import os
import json
import re

class CameraSource:
    def __init__(self, config:dict):

        self.__config = config

        # ============== CONNECT TO SURVEILLANCE STATION ==============
    def connect(self):
        """Establish connection to Synology Surveillance Station."""
        print("Connecting to Synology Surveillance Station...")
        
        ss = surveillancestation.SurveillanceStation(
            ip_address = self.__config["ip_address"],
            port = self.__config["port"],
            username = self.__config["username"],
            password = self.__config["password"],
            secure = self.__config["secure"],
            cert_verify = self.__config["cert_verify"],
            dsm_version = self.__config["dsm_version"],
            otp_code = self.__config["otp_code"],
            debug=True
        )

        cameras = ss.camera_list()['data']['cameras']
        for camera in cameras:
            print(f'{camera['id']} {camera['ip']} {camera['newName']}\n')
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

        with open(os.path.join(outputDir,f'{camera['name']}.jpg'), 'wb') as file:
            file.write(snap_shot)
        
        obj = self.getPath(ss, camera_id)
        print(json.dumps(obj))
        return obj
    
    def getPath(self, ss:surveillancestation.SurveillanceStation, cameraId:int):
        camera_object = None
        for camera in ss.get_live_path(cameraId)['data']:
            #Replaces the ip address with the dns entry
            for key in camera.keys():
                if(key.endswith("Path")):
                    camera[key] = self.fixAddress(self.__config["ip_address"], camera[key])
            camera_object = camera                    

        return camera_object
        

    def fixAddress(self, dns:str, url:str):
        """The server doesn't use the define security configuration and must be modified to support it.

        Args:
            dns (_type_): server dns to replace the ip address with
            url (_type_): url to modify

        Returns:
            _type_: _description_
        """
        temp = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', dns, url)
        temp = temp.replace(':5000', f":{self.__config['port']}")
        if(self.__config['secure']):
            temp = temp.replace('http://', "https://")
        return temp
