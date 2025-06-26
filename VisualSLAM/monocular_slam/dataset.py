# dataset.py
import os
from PIL import Image # Ensure PIL.Image is imported here as it's used directly
import torch # Still might be used for transforms, even if not directly in this example
from torch.utils.data import Dataset
import bisect

class TUMTimeDataset(Dataset):
    def __init__(self, rgb_file, depth_file, transform=None, depth_transform=None, base_dir=""):
        self.base_dir = base_dir

        self.rgb_data_raw = self._read_file(os.path.join(base_dir, rgb_file))
        self.depth_data_raw = self._read_file(os.path.join(base_dir, depth_file))

        self.event_list = []
        for timestamp, paths in self.rgb_data_raw:
            self.event_list.append((timestamp, 'rgb', paths[0], os.path.basename(paths[0])))
        for timestamp, paths in self.depth_data_raw:
            self.event_list.append((timestamp, 'depth', paths[0], os.path.basename(paths[0])))

        self.event_list.sort(key=lambda x: x[0])

        if self.event_list:
            self.min_overall_timestamp = self.event_list[0][0]
            self.event_list = [
                (timestamp - self.min_overall_timestamp, event_type, relative_path, filename)
                for timestamp, event_type, relative_path, filename in self.event_list
            ]
        else:
            self.min_overall_timestamp = 0.0
            print("Warning: No RGB or Depth data found in the files.")

        self.normalized_timestamps = [event[0] for event in self.event_list]

        self.transform = transform
        self.depth_transform = depth_transform
        print(f"Dataset loaded. Total {len(self.event_list)} chronological events (RGB and Depth only).")
        print(f"Timestamps normalized. First event at time 0.0.")

    def _read_file(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        data = []
        for line in lines:
            parts = line.replace(",", " ").replace("\t", " ").split()
            timestamp = float(parts[0])
            data.append((timestamp, parts[1:]))
        return data

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, idx):
        normalized_timestamp, event_type, relative_path, filename = self.event_list[idx]

        result = {
            "timestamp": normalized_timestamp,
            "event_type": event_type,
            "filename": filename
        }

        if event_type == 'rgb':
            rgb_full_path = os.path.join(self.base_dir, relative_path)
            rgb = Image.open(rgb_full_path).convert("RGB")
            if self.transform:
                rgb = self.transform(rgb)
            result["rgb"] = rgb
        elif event_type == 'depth':
            depth_full_path = os.path.join(self.base_dir, relative_path)
            depth = Image.open(depth_full_path).convert("I") # TUM depth is 16-bit
            if self.depth_transform:
                depth = self.depth_transform(depth)
                if isinstance(depth, torch.Tensor) and depth.ndim == 2:
                    depth = depth.unsqueeze(0) # Add channel dim for consistency if needed
            result["depth"] = depth
        return result

    def get_data_at_timestamp(self, desired_normalized_timestamp, tolerance=0.01):
        if not self.event_list:
            print("Dataset is empty. Cannot retrieve data by timestamp.")
            return None

        idx = bisect.bisect_left(self.normalized_timestamps, desired_normalized_timestamp)

        closest_idx = -1
        min_time_diff = float('inf')

        if idx < len(self.normalized_timestamps):
            diff = abs(self.normalized_timestamps[idx] - desired_normalized_timestamp)
            if diff < min_time_diff:
                min_time_diff = diff
                closest_idx = idx

        if idx > 0:
            diff = abs(self.normalized_timestamps[idx - 1] - desired_normalized_timestamp)
            if diff < min_time_diff:
                min_time_diff = diff
                closest_idx = idx - 1

        if closest_idx != -1 and min_time_diff <= tolerance:
            return self.__getitem__(closest_idx)
        else:
            return None