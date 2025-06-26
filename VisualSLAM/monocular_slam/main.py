# main.py
import os
from PIL import Image # PIL is still needed by TUMTimeDataset to open images
from dataset import TUMTimeDataset
from slam_interface import SlamInterface # Import the new interface
from slam_system import SLAMSystem     # Import the core SLAM system
from event_manager import EventManager
import time

if __name__ == "__main__":
    # Define the base directory for your TUM dataset
    DATASET_BASE_DIR = r'rgbd_dataset_freiburg1_desk'
    RGB_FILE = r'rgb.txt'
    DEPTH_FILE = r'depth.txt'

    try:
        print(f"Loading dataset from: {DATASET_BASE_DIR}")
        dataset = TUMTimeDataset(
            rgb_file=RGB_FILE,
            depth_file=DEPTH_FILE,
            base_dir=DATASET_BASE_DIR
        )

        # 2. Instantiate the core SLAM System
        # Pass your actual camera intrinsics here if you have them calibrated.
        # Otherwise, the default dummy intrinsics will be used.
        my_slam_system_core = SLAMSystem(name="MyAwesomeSLAM_Core")

        # 3. Instantiate the SlamInterface, passing the core SLAM system to it
        my_slam_interface = SlamInterface(slam_system_instance=my_slam_system_core)

        # 4. Initialize the Event Manager
        # time_scale_factor: 1.0 for real-time, 2.0 for 2x speed, 0.5 for half speed
        event_manager = EventManager(dataset=dataset, time_scale_factor=2.0) # Example: 2x speed

        # 5. Register the interface with the Event Manager
        event_manager.register_listener("slam_interface", my_slam_interface)

        # 6. Run the simulation
        event_manager.run()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Please ensure the TUM dataset directory '{DATASET_BASE_DIR}' exists")
        print(f"and contains '{RGB_FILE}' and '{DEPTH_FILE}' files.")
        print(f"Also, verify that the image paths inside '{RGB_FILE}' and '{DEPTH_FILE}'")
        print(f"are correct relative to '{DATASET_BASE_DIR}'.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    finally:
        print("\nSimulation finished.")
        # Ensure Open3D visualizer is destroyed properly if it was created
        if 'my_slam_system_core' in locals() and hasattr(my_slam_system_core, 'vis'):
            my_slam_system_core.vis.destroy_window()
            print("Open3D visualizer window closed.")