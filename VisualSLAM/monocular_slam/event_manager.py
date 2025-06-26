# event_manager.py
import time
from dataset import TUMTimeDataset # Already correct

class EventManager:
    """
    Manages the dispatch of sensor data from TUMTimeDataset to registered listeners
    based on a simulated time progression.
    """
    def __init__(self, dataset: TUMTimeDataset, time_scale_factor=1.0):
        """
        Initializes the EventManager.

        Args:
            dataset (TUMTimeDataset): The dataset containing the chronological events.
            time_scale_factor (float): Controls the simulation speed.
                                       1.0 means real-time playback (relative to dataset timestamps).
                                       0.5 means half speed, 2.0 means double speed.
        """
        self.dataset = dataset
        self.listeners = {}
        self.time_scale_factor = time_scale_factor
        self.current_sim_time = 0.0
        self.event_pointer = 0

        if self.dataset and self.dataset.event_list:
            self.total_duration = self.dataset.event_list[-1][0]
        else:
            self.total_duration = 0.0

        print(f"EventManager initialized. Simulation speed factor: {self.time_scale_factor}x")

    def register_listener(self, name: str, listener):
        """Registers a listener to receive data events."""
        # Updated check for the new method names in SlamInterface
        if not hasattr(listener, 'on_rgb_data') or not hasattr(listener, 'on_depth_data'):
            raise ValueError("Listener must have 'on_rgb_data' and 'on_depth_data' methods.")
        self.listeners[name] = listener
        print(f"Listener '{name}' registered.")

    def unregister_listener(self, name: str):
        """Unregisters a listener."""
        if name in self.listeners:
            del self.listeners[name]
            print(f"Listener '{name}' unregistered.")
        else:
            print(f"Listener '{name}' not found.")

    def run(self, max_duration=None):
        """
        Starts the event simulation.

        Args:
            max_duration (float, optional): Maximum normalized time to simulate.
                                            If None, simulates until all events are processed.
        """
        start_real_time = time.time()

        self.current_sim_time = 0.0
        self.event_pointer = 0

        print("\n--- Starting Event Simulation ---")
        if self.total_duration > 0:
            print(f"Total dataset duration: {self.total_duration:.2f} seconds (normalized).")
        else:
            print("No events to simulate.")
            return

        while self.event_pointer < len(self.dataset.event_list):
            event_timestamp, event_type, _, _ = self.dataset.event_list[self.event_pointer]

            if max_duration is not None and event_timestamp > max_duration:
                print(f"Reached max_duration {max_duration:.2f}. Stopping simulation.")
                break

            time_to_wait = (event_timestamp - self.current_sim_time) / self.time_scale_factor

            if time_to_wait > 0:
                time.sleep(time_to_wait)

            self.current_sim_time = event_timestamp

            event_data = self.dataset[self.event_pointer]

            if event_type == 'rgb':
                for listener_name, listener in self.listeners.items():
                    listener.on_rgb_data(event_data) # Call the new method name
            elif event_type == 'depth':
                for listener_name, listener in self.listeners.items():
                    listener.on_depth_data(event_data) # Call the new method name

            self.event_pointer += 1

            if self.total_duration > 0:
                progress = (self.current_sim_time / self.total_duration) * 100
                print(f"Sim Time: {self.current_sim_time:.6f} / {self.total_duration:.2f} ({progress:.1f}%)")
            else:
                print(f"Sim Time: {self.current_sim_time:.6f}")


        end_real_time = time.time()
        print(f"\n--- Event Simulation Complete ---")
        print(f"Total simulated duration: {self.current_sim_time:.2f} seconds")
        print(f"Total real time taken: {end_real_time - start_real_time:.2f} seconds")