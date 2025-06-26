# slam_interface.py

import numpy as np
import cv2 

class SlamInterface:
    """
    Acts as the interface between the EventManager and the core SLAMSystem.
    Receives RGB and Depth data and passes it to the SLAMSystem for processing.
    """
    def __init__(self, slam_system_instance, name="SLAM Interface"):
        self.name = name
        if not hasattr(slam_system_instance, 'process_rgb_frame') or \
           not hasattr(slam_system_instance, 'process_depth_frame'):
            raise ValueError("SLAM system instance must have 'process_rgb_frame' and 'process_depth_frame' methods.")
        self.slam_system = slam_system_instance # Reference to the actual SLAM system
        self._last_rgb_timestamp = -1
        self._last_depth_timestamp = -1
        print(f"{self.name} initialized, connected to {self.slam_system.name}.")

    def on_rgb_data(self, data):
        timestamp = data["timestamp"]
        rgb_pil_image = data["rgb"]
        rgb_np_array = np.array(rgb_pil_image)
        rgb_gray = cv2.cvtColor(rgb_np_array, cv2.COLOR_RGB2GRAY) # Convert to grayscale for SLAM

        self.slam_system.process_rgb_frame(timestamp, rgb_gray)
        self._last_rgb_timestamp = timestamp

    def on_depth_data(self, data):
        timestamp = data["timestamp"]
        depth_pil_image = data["depth"]
        depth_np_array = np.array(depth_pil_image)

        # TUM depth images are often 16-bit grayscale where values are in millimeters.
        # Common scale factor for TUM is 5000, meaning value 5000 = 1 meter.
        # You'll need to confirm the scale factor for your specific TUM dataset.
        depth_in_meters = depth_np_array.astype(np.float32) / 5000.0

        self.slam_system.process_depth_frame(timestamp, depth_in_meters)
        self._last_depth_timestamp = timestamp

    def on_event_complete(self, event_type, timestamp):
        # This callback is usually not used for frame-by-frame SLAM processing
        pass