# slam_system.py

import cv2
import numpy as np
import open3d as o3d
from collections import deque # For buffering frames
import time # For internal timing if needed
import matplotlib.pyplot as plt # For 2D visualization of trajectory

# (Optional) Example for camera intrinsics - you'd get this from your dataset's calibration
# This is a dummy example; replace with your actual camera intrinsics
CAMERA_FX = 525.0
CAMERA_FY = 525.0
CAMERA_CX = 319.5
CAMERA_CY = 239.5
CAMERA_INTRINSICS = np.array([
    [CAMERA_FX, 0, CAMERA_CX],
    [0, CAMERA_FY, CAMERA_CY],
    [0, 0, 1]
], dtype=np.float32)


class SLAMSystem:
    """
    Implements the core SLAM algorithms: Initialization, Tracking, Local Mapping,
    Loop Closure, and Global Optimization (Bundle Adjustment, Pose Graph Optimization).
    """
    def __init__(self, name="Core SLAM System", camera_intrinsics=CAMERA_INTRINSICS):
        self.name = name
        print(f"{self.name} initialized.")

        # --- SLAM State Variables ---
        self.camera_intrinsics = camera_intrinsics
        self.feature_detector = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.keyframes = []
        self.map_points = {}
        
        self.last_rgb_gray = None
        self.last_depth_image = None
        self.last_keypoints = None
        self.last_descriptors = None
        self.last_pose_matrix = np.eye(4) # Current estimated camera pose (World to Camera)

        # Buffers for incoming sensor data, awaiting synchronization
        self.rgb_buffer = deque()
        self.depth_buffer = deque()
        self.sync_tolerance = 0.01

        # --- For Visualization (Open3D) ---
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=f"{self.name} SLAM Viewer (3D)", width=960, height=540)
        self.map_point_cloud = o3d.geometry.PointCloud()
        self.trajectory_line_set = o3d.geometry.LineSet()
        
        # Add trajectory and map point cloud to visualizer only ONCE
        self.vis.add_geometry(self.trajectory_line_set)
        self.vis.add_geometry(self.map_point_cloud)
        
        # FIX HERE: Use PointCloud for current camera position
        self.current_camera_marker = o3d.geometry.PointCloud()
        self.current_camera_marker.points = o3d.utility.Vector3dVector([[0, 0, 0]]) # Initialize at origin
        self.current_camera_marker.colors = o3d.utility.Vector3dVector([[0, 0.7, 0]]) # Green
        self.vis.add_geometry(self.current_camera_marker)


        self.vis_render_options = self.vis.get_render_option()
        self.vis_render_options.point_size = 10.0 # Make the point larger to be visible
        self.vis_render_options.background_color = np.asarray([0.0, 0.0, 0.0])

        print(f"{self.name} Open3D Visualizer initialized.")

        # --- For Visualization (Matplotlib) ---
        plt.ion() # Turn on interactive mode for non-blocking plots
        self.fig_mpl, self.ax_mpl = plt.subplots(figsize=(6, 6))
        self.ax_mpl.set_title("SLAM Trajectory (X-Z Plane)")
        self.ax_mpl.set_xlabel("X (meters)")
        self.ax_mpl.set_ylabel("Z (meters)") # Assuming Z is forward
        self.ax_mpl.grid(True)
        self.mpl_line, = self.ax_mpl.plot([], [], 'r-') # Initialize an empty line object
        self.mpl_scatter, = self.ax_mpl.plot([], [], 'bo') # For current position
        
        self.trajectory_history = [] # To store (x, y, z) coordinates for plotting

        print(f"{self.name} Matplotlib Visualizer initialized.")
        print("Note: Complex SLAM components (Local Mapping, Loop Closure, Optimization) are placeholders.")


    def process_rgb_frame(self, timestamp, rgb_gray_image):
        """Receives and buffers an RGB frame."""
        self.rgb_buffer.append((timestamp, rgb_gray_image))
        self._try_process_synchronized_frame()
        self._update_visualization() # Update visualization more frequently, even if no SLAM update

    def process_depth_frame(self, timestamp, depth_image_in_meters):
        """Receives and buffers a Depth frame."""
        self.depth_buffer.append((timestamp, depth_image_in_meters))
        self._try_process_synchronized_frame()
        self._update_visualization() # Update visualization more frequently, even if no SLAM update

    def _try_process_synchronized_frame(self):
        """
        Attempts to find a synchronized RGB-D pair from buffers and process it.
        This is a simple nearest-neighbor synchronization. A more robust system
        might use a fixed-size window or event-driven sync.
        """
        # Keep processing as long as there are potential pairs
        while self.rgb_buffer and self.depth_buffer:
            rgb_ts, rgb_img = self.rgb_buffer[0]
            depth_ts, depth_img = self.depth_buffer[0]

            if abs(rgb_ts - depth_ts) < self.sync_tolerance:
                # Found a synchronized pair
                self.rgb_buffer.popleft()
                self.depth_buffer.popleft()
                self._process_single_frame(rgb_ts, rgb_img, depth_img)
            elif rgb_ts < depth_ts:
                # RGB is older. If its corresponding depth hasn't arrived within tolerance, discard it.
                if depth_ts - rgb_ts > self.sync_tolerance:
                    print(f"[{self.name}] Warning: Discarding old RGB frame at {rgb_ts:.6f} (no matching depth within tolerance).")
                    self.rgb_buffer.popleft()
                else:
                    # RGB is older but still within tolerance, wait for depth to catch up
                    break
            else: # depth_ts < rgb_ts
                # Depth is older. If its corresponding RGB hasn't arrived within tolerance, discard it.
                if rgb_ts - depth_ts > self.sync_tolerance:
                    print(f"[{self.name}] Warning: Discarding old Depth frame at {depth_ts:.6f} (no matching RGB within tolerance).")
                    self.depth_buffer.popleft()
                else:
                    # Depth is older but still within tolerance, wait for RGB to catch up
                    break


    def _process_single_frame(self, timestamp, rgb_gray, depth_image):
        """
        Main SLAM processing logic for a single synchronized RGB-D frame.
        """
        # --- 1. Feature Detection and Description ---
        keypoints, descriptors = self.feature_detector.detectAndCompute(rgb_gray, None)

        if keypoints is None or descriptors is None or len(keypoints) == 0:
            print(f"[{self.name}] No features detected in frame at {timestamp:.6f}")
            # Still update last frame's data even if no new features,
            # but don't attempt pose estimation or map update for this frame.
            self.last_rgb_gray = rgb_gray
            self.last_depth_image = depth_image
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            return # Exit early if no features

        # --- 2. Tracking (Visual Odometry / Pose Estimation) ---
        pose_estimated_this_frame = False # Flag to track if pose was updated

        if self.last_keypoints is not None and self.last_descriptors is not None:
            matches = self.matcher.match(self.last_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:100] 

            if len(good_matches) > 8:
                pts1_3d = self._get_3d_points_from_depth(self.last_keypoints, self.last_depth_image, self.camera_intrinsics)
                
                valid_matches = []
                valid_pts1_3d = [] # 3D points from previous frame (in previous camera's coordinate system)
                valid_pts2 = []    # 2D points from current frame (matched features)
                
                for m in good_matches:
                    if not np.isnan(pts1_3d[m.queryIdx]).any():
                        u_curr, v_curr = int(keypoints[m.trainIdx].pt[0]), int(keypoints[m.trainIdx].pt[1])
                        if 0 <= u_curr < depth_image.shape[1] and 0 <= v_curr < depth_image.shape[0]:
                            z_curr = depth_image[v_curr, u_curr]
                            if z_curr > 0.0 and z_curr < 10.0: # Valid depth range
                                valid_matches.append(m)
                                valid_pts1_3d.append(pts1_3d[m.queryIdx])
                                valid_pts2.append(keypoints[m.trainIdx].pt)
                
                valid_pts1_3d = np.array(valid_pts1_3d, dtype=np.float32)
                valid_pts2 = np.array(valid_pts2, dtype=np.float32)

                if len(valid_pts1_3d) >= 4:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        valid_pts1_3d, valid_pts2, self.camera_intrinsics, None,
                        reprojectionError=4.0
                    )
                    
                    if success:
                        R_rel, _ = cv2.Rodrigues(rvec)
                        T_prev_to_curr = np.eye(4)
                        T_prev_to_curr[:3, :3] = R_rel
                        T_prev_to_curr[:3, 3] = tvec.flatten()

                        # Update global pose: World_T_Current = World_T_Last @ Last_T_Current
                        self.last_pose_matrix = self.last_pose_matrix @ T_prev_to_curr
                        pose_estimated_this_frame = True # Mark as successful
                        print(f"[{self.name}] Frame {timestamp:.6f}: Pose T = {self.last_pose_matrix[:3, 3]} (Matches: {len(valid_matches)})")

                    else:
                        print(f"[{self.name}] PnP failed for frame at {timestamp:.6f}. Tracking lost.")
                else:
                    print(f"[{self.name}] Not enough valid 3D-2D matches ({len(valid_pts1_3d)}) for PnP at {timestamp:.6f}. Tracking lost.")
            else:
                print(f"[{self.name}] Not enough good matches ({len(good_matches)}) for tracking at {timestamp:.6f}. Tracking lost.")
        else:
            print(f"[{self.name}] Initializing SLAM with first frame at {timestamp:.6f}.")
            pose_estimated_this_frame = True # First frame, pose is identity by default

        # Update last frame's data for the next iteration
        self.last_rgb_gray = rgb_gray
        self.last_depth_image = depth_image
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors

        # --- NEW SECTION: Map Point Generation & Update ---
        if pose_estimated_this_frame: # Only add points if we have a valid pose for the current frame
            # 1. Get 3D points from current frame's keypoints and depth
            # These points are in the current camera's coordinate system.
            current_frame_3d_points_cam = self._get_3d_points_from_depth(
                keypoints, depth_image, self.camera_intrinsics
            )
            
            # Filter out points that couldn't be validly projected (NaNs)
            valid_3d_points_cam = current_frame_3d_points_cam[~np.isnan(current_frame_3d_points_cam).any(axis=1)]

            if len(valid_3d_points_cam) > 0:
                # 2. Transform these 3D points from current camera frame to World frame
                # World_P = World_T_Camera @ Camera_P (in homogeneous coordinates)
                homogeneous_points_cam = np.hstack((valid_3d_points_cam, np.ones((len(valid_3d_points_cam), 1))))
                
                # self.last_pose_matrix is World_T_CurrentCamera
                homogeneous_points_world = (self.last_pose_matrix @ homogeneous_points_cam.T).T
                
                # Extract 3D points from homogeneous coordinates
                new_world_points = homogeneous_points_world[:, :3]

                # 3. Add these new points to the global map point cloud
                # For simplicity, we'll just append them. In a real SLAM system,
                # you'd implement more sophisticated data association, merging,
                # and outlier rejection.
                
                # Get existing points (if any)
                existing_points_np = np.asarray(self.map_point_cloud.points)
                
                # Concatenate new points with existing points
                all_points_np = np.vstack((existing_points_np, new_world_points))
                
                # Update the map_point_cloud geometry
                self.map_point_cloud.points = o3d.utility.Vector3dVector(all_points_np)
                
                # Optionally, set colors for the new points (e.g., from RGB image)
                # For simplicity, let's just make them a consistent grey/white
                # Ensure colors array matches the new number of points
                self.map_point_cloud.colors = o3d.utility.Vector3dVector(np.tile([0.8, 0.8, 0.8], (len(all_points_np), 1)))

                # print(f"[{self.name}] Added {len(new_world_points)} points to map. Total map points: {len(all_points_np)}")


        # --- 4. Loop Closure (Asynchronous - runs periodically) ---
        # ... (placeholders) ...


    def _get_3d_points_from_depth(self, keypoints, depth_image, K):
        """
        Projects 2D keypoints to 3D points using depth image and camera intrinsics.
        Returns a NumPy array of (N, 3) 3D points (in camera's coordinate system).
        Points with invalid depth will have NaN coordinates.
        """
        points_3d = np.full((len(keypoints), 3), np.nan, dtype=np.float32)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        for i, kp in enumerate(keypoints):
            u, v = int(kp.pt[0]), int(kp.pt[1])

            # Ensure coordinates are within image bounds
            if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                z = depth_image[v, u] # Depth value at (v, u) in meters

                # Check for valid depth (e.g., > 0 and not excessively far)
                if z > 0.0 and z < 10.0: # Filter out 0 depth (no data) or excessively far points
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points_3d[i] = [x, y, z]
        return points_3d

    def _update_visualization(self):
        """
        Updates both Open3D and Matplotlib visualizations.
        Called frequently to keep viewers responsive.
        """
        # Ensure that self.trajectory_history has been populated
        # This occurs in _process_single_frame, but visualization updates
        # more often. Only update if there's actually new pose data.
        current_pose_translation = self.last_pose_matrix[:3, 3] # Current camera position in world coordinates
        if len(self.trajectory_history) == 0 or not np.allclose(self.trajectory_history[-1], current_pose_translation):
            self.trajectory_history.append(current_pose_translation.copy())


        # --- Open3D Visualization ---
        o3d_trajectory_points = np.array(self.trajectory_history)
        
        if len(o3d_trajectory_points) > 1:
            points_o3d = o3d.utility.Vector3dVector(o3d_trajectory_points)
            
            lines = []
            for i in range(len(o3d_trajectory_points) - 1):
                lines.append([i, i+1])
            
            colors = [[1, 0, 0] for _ in range(len(lines))] # Red line for trajectory
            
            self.trajectory_line_set.points = points_o3d
            self.trajectory_line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
            self.trajectory_line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            self.vis.update_geometry(self.trajectory_line_set)

            # --- CORRECTED CAMERA MARKER UPDATE ---
            self.current_camera_marker.points = o3d.utility.Vector3dVector([o3d_trajectory_points[-1]])
            self.vis.update_geometry(self.current_camera_marker)
        elif len(o3d_trajectory_points) == 1: # Handle the very first point
            self.current_camera_marker.points = o3d.utility.Vector3dVector([o3d_trajectory_points[-1]])
            self.vis.update_geometry(self.current_camera_marker)
            
        # --- IMPORTANT: Update the map point cloud ---
        self.vis.update_geometry(self.map_point_cloud)

        # Adjust camera view to fit all geometries - CRITICAL for seeing something
        self.vis.reset_view_point(True) # This now has points to fit to if map is populated

        self.vis.poll_events()
        self.vis.update_renderer()

        # --- Matplotlib Visualization ---
        if len(self.trajectory_history) > 0: # Check if trajectory_history is not empty
            traj_np = np.array(self.trajectory_history)
            
            # Plot X (forward) vs Z (depth/up/down depending on coordinate system)
            x_coords = traj_np[:, 0]
            z_coords = traj_np[:, 2] 

            self.mpl_line.set_data(x_coords, z_coords)
            self.mpl_scatter.set_data([x_coords[-1]], [z_coords[-1]]) # Current position dot

            # Dynamically adjust plot limits
            self.ax_mpl.relim()
            self.ax_mpl.autoscale_view()
            
            # Draw the plot
            self.fig_mpl.canvas.draw()
            self.fig_mpl.canvas.flush_events() 
        else:
            # For the very first point, set the scatter data and initial limits
            if self.trajectory_history: # This condition is essentially same as above, but here for clarity.
                current_pos = np.array(self.trajectory_history[-1])
                self.mpl_scatter.set_data([current_pos[0]], [current_pos[2]])
                # Set initial limits around the first point for visibility
                self.ax_mpl.set_xlim(current_pos[0] - 1, current_pos[0] + 1)
                self.ax_mpl.set_ylim(current_pos[2] - 1, current_pos[2] + 1)
                self.fig_mpl.canvas.draw()
                self.fig_mpl.canvas.flush_events()

    def _add_keyframe(self, rgb_gray, depth_image, keypoints, descriptors, pose):
        """
        Decides if the current frame is a new keyframe based on movement heuristics
        and adds it to the keyframe database.
        A 'Keyframe' object would typically store:
        - Its image data (or path to it)
        - Its estimated camera pose
        - Its detected keypoints and descriptors
        - References to observed 3D map points
        """
        print(f"[{self.name}] Placeholder: Adding keyframe logic.")
        pass

    def _triangulate_new_points(self):
        """
        Identifies new 3D points by matching features across new keyframes
        and existing map points, then triangulates their 3D positions.
        """
        print(f"[{self.name}] Placeholder: Triangulating new map points.")
        pass

    def _perform_local_bundle_adjustment(self):
        """
        Optimizes a sliding window of recent keyframes and their associated map points
        to improve their consistency. This is usually a non-linear optimization problem.
        """
        print(f"[{self.name}] Placeholder: Performing local Bundle Adjustment.")
        pass

    def _check_for_loop_closure(self):
        """
        Compares the current keyframe to a database of past keyframes (e.g., using
        Bag-of-Words for image retrieval) to detect if a previously visited location
        has been revisited.
        """
        print(f"[{self.name}] Placeholder: Checking for loop closure.")
        pass

    def _perform_global_optimization(self):
        """
        If a loop closure is detected and verified, this function triggers a global
        optimization (either Bundle Adjustment or Pose Graph Optimization) over the
        entire map and trajectory to correct accumulated drift.
        """
        print(f"[{self.name}] Placeholder: Performing global optimization (BA/PGO).")
        pass