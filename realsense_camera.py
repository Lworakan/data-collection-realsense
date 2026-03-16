"""
Intel RealSense D400 Series Camera Capture Module

Provides a RealSenseCamera class for capturing aligned color + depth frames
from Intel RealSense D415/D435 cameras. Designed for integration with the
data_collection app alongside phone cam and notebook cam sources.

Outputs per frame:
    1. Color image (PNG)
    2. Colorized depth image (PNG)
    3. Raw depth array (NPY, float32 meters)
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class RealSenseCamera:
    """Intel RealSense D400 series camera capture with aligned color + depth.

    Args:
        width: Stream width in pixels (default: 848).
        height: Stream height in pixels (default: 480).
        fps: Frames per second (default: 30).
        enable_filters: Apply post-processing filters to depth (default: True).
    """

    def __init__(self, width: int = 848, height: int = 480, fps: int = 30,
                 enable_filters: bool = True):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_filters = enable_filters

        self._pipeline = None
        self._align = None
        self._colorizer = None
        self._filters = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Initialize and start the RealSense pipeline."""
        if rs is None:
            raise ImportError(
                "pyrealsense2 is not installed. "
                "Install it with: pip install pyrealsense2"
            )
        if self._running:
            return

        self._pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)

        profile = self._pipeline.start(config)

        # Print device info
        device = profile.get_device()
        print(f"[RealSense] Device: {device.get_info(rs.camera_info.name)}")
        print(f"[RealSense] Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"[RealSense] Resolution: {self.width}x{self.height} @ {self.fps}fps")

        # Depth-to-color alignment
        self._align = rs.align(rs.stream.color)

        # Depth colorizer for visualization
        self._colorizer = rs.colorizer()
        self._colorizer.set_option(rs.option.color_scheme, 0)  # Jet colormap

        # Post-processing filters
        if self.enable_filters:
            self._filters = [
                rs.decimation_filter(),
                rs.spatial_filter(),
                rs.temporal_filter(),
                rs.hole_filling_filter(),
            ]
            # Decimation at factor 1 = no downscale, just smoothing metadata
            self._filters[0].set_option(rs.option.filter_magnitude, 1)

        # Let auto-exposure stabilize
        for _ in range(30):
            self._pipeline.wait_for_frames()

        self._running = True
        print("[RealSense] Pipeline started.")

    def stop(self):
        """Stop the RealSense pipeline and release resources."""
        if self._running and self._pipeline:
            self._pipeline.stop()
            self._running = False
            self._pipeline = None
            self._align = None
            self._colorizer = None
            self._filters = []
            print("[RealSense] Pipeline stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_frame(self) -> dict:
        """Capture a single aligned color + depth frame.

        Returns:
            dict with keys:
                - color_image: np.ndarray (H, W, 3) uint8 BGR
                - depth_image: np.ndarray (H, W, 3) uint8 BGR colorized depth
                - depth_raw:   np.ndarray (H, W) float32 depth in meters
                - timestamp:   float (epoch seconds)
        """
        if not self._running:
            raise RuntimeError("Pipeline not started. Call start() first.")

        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture frames.")

        # Apply post-processing filters to depth
        if self.enable_filters:
            for f in self._filters:
                depth_frame = f.process(depth_frame)

        # Color image (BGR)
        color_image = np.asanyarray(color_frame.get_data())

        # Raw depth in meters (float32)
        depth_scale = (aligned.get_device()
                       if hasattr(aligned, 'get_device') else
                       self._pipeline.get_active_profile()
                       .get_device()
                       .first_depth_sensor()
                       .get_depth_scale())
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

        # Colorized depth for visualization
        depth_colorized = self._colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(depth_colorized.get_data())

        return {
            "color_image": color_image,
            "depth_image": depth_image,
            "depth_raw": depth_raw,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def save_frame(frame_data: dict, save_dir: str, frame_id: str):
        """Save captured frame data to disk.

        Saves to three subfolders:
            color/     {frame_id}.png  — BGR color image
            depth/     {frame_id}.png  — colorized depth visualization
            depth_raw/ {frame_id}.npy  — raw depth in meters (float32)

        Args:
            frame_data: dict returned by capture_frame().
            save_dir: session directory (subfolders created automatically).
            frame_id: filename, e.g. "000042".
        """
        color_dir = os.path.join(save_dir, "color")
        depth_dir = os.path.join(save_dir, "depth")
        raw_dir = os.path.join(save_dir, "depth_raw")

        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)

        cv2.imwrite(os.path.join(color_dir, f"{frame_id}.png"), frame_data["color_image"])
        cv2.imwrite(os.path.join(depth_dir, f"{frame_id}.png"), frame_data["depth_image"])
        np.save(os.path.join(raw_dir, f"{frame_id}.npy"), frame_data["depth_raw"])

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview(self):
        """Live preview window showing color and depth side-by-side.

        Press 'q' to quit, 's' to save a single frame to ./datasets/realsense/.
        """
        if not self._running:
            self.start()

        print("[RealSense] Preview started. Press 'q' to quit, 's' to save frame.")
        snap_count = 0

        while True:
            frame_data = self.capture_frame()

            color = frame_data["color_image"]
            depth = frame_data["depth_image"]

            # Resize depth to match color if shapes differ after filtering
            if depth.shape[:2] != color.shape[:2]:
                depth = cv2.resize(depth, (color.shape[1], color.shape[0]))

            combined = np.hstack([color, depth])
            cv2.imshow("RealSense | Color + Depth (press q to quit)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_dir = os.path.join("datasets", "realsense", "snapshots")
                fid = f"{snap_count:06d}"
                self.save_frame(frame_data, save_dir, fid)
                print(f"[RealSense] Saved snapshot {fid} to {save_dir}/")
                snap_count += 1

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Collect (2 windows + save)
    # ------------------------------------------------------------------

    def collect(self, save_dir: str = None, max_frames: int = None,
                duration_sec: float = None):
        """Collect data with 2 live windows (color + depth) while saving to disk.

        Opens two separate OpenCV windows for color and depth, and saves
        every frame (color PNG, depth PNG, depth_raw NPY) to disk.

        Press 'q' to stop. Press 'p' to pause/resume saving.

        Args:
            save_dir: output directory (default: datasets/realsense/{timestamp}/).
            max_frames: stop after N frames (None = unlimited).
            duration_sec: stop after N seconds (None = unlimited).
        """
        if not self._running:
            self.start()

        if save_dir is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join("datasets", "realsense", session_id)

        os.makedirs(save_dir, exist_ok=True)

        # Window names
        win_color = "RealSense - Color"
        win_depth = "RealSense - Depth"
        cv2.namedWindow(win_color, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(win_depth, cv2.WINDOW_AUTOSIZE)

        print(f"[RealSense] Collecting to {save_dir}/")
        print("[RealSense] Press 'q' to stop, 'p' to pause/resume saving.")

        frame_count = 0
        paused = False
        t_start = time.time()

        try:
            while True:
                frame_data = self.capture_frame()
                color = frame_data["color_image"]
                depth = frame_data["depth_image"]

                # Resize depth to match color if shapes differ after filtering
                if depth.shape[:2] != color.shape[:2]:
                    depth = cv2.resize(depth, (color.shape[1], color.shape[0]))

                # Add status text overlay
                elapsed = time.time() - t_start
                status = f"Frame: {frame_count} | {elapsed:.1f}s"
                if paused:
                    status += " | PAUSED"
                cv2.putText(color, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(depth, f"Depth | {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show in 2 separate windows
                cv2.imshow(win_color, color)
                cv2.imshow(win_depth, depth)

                # Save frame (use original data without overlay)
                if not paused:
                    fid = f"{frame_count:06d}"
                    self.save_frame(frame_data, save_dir, fid)
                    frame_count += 1

                    if frame_count % 30 == 0:
                        print(f"  [{frame_count} frames, {elapsed:.1f}s]")

                    if max_frames and frame_count >= max_frames:
                        break
                    if duration_sec and elapsed >= duration_sec:
                        break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    state = "PAUSED" if paused else "RESUMED"
                    print(f"[RealSense] {state} at frame {frame_count}")

        except KeyboardInterrupt:
            print("\n[RealSense] Collection interrupted.")

        cv2.destroyAllWindows()
        elapsed = time.time() - t_start
        print(f"[RealSense] Saved {frame_count} frames in {elapsed:.1f}s "
              f"({frame_count / max(elapsed, 1e-6):.1f} fps) to {save_dir}/")

    # ------------------------------------------------------------------
    # Continuous recording (headless)
    # ------------------------------------------------------------------

    def record(self, save_dir: str = None, max_frames: int = None,
               duration_sec: float = None):
        """Record frames continuously to disk (no GUI windows).

        Args:
            save_dir: output directory (default: datasets/realsense/{timestamp}/).
            max_frames: stop after N frames (None = unlimited).
            duration_sec: stop after N seconds (None = unlimited).
                          If both max_frames and duration_sec are None,
                          records until KeyboardInterrupt.
        """
        if not self._running:
            self.start()

        if save_dir is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join("datasets", "realsense", session_id)

        os.makedirs(save_dir, exist_ok=True)
        print(f"[RealSense] Recording to {save_dir}/")

        frame_count = 0
        t_start = time.time()

        try:
            while True:
                frame_data = self.capture_frame()
                fid = f"{frame_count:06d}"
                self.save_frame(frame_data, save_dir, fid)
                frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{frame_count} frames, {elapsed:.1f}s]")

                if max_frames and frame_count >= max_frames:
                    break
                if duration_sec and (time.time() - t_start) >= duration_sec:
                    break

        except KeyboardInterrupt:
            print("\n[RealSense] Recording interrupted.")

        elapsed = time.time() - t_start
        print(f"[RealSense] Saved {frame_count} frames in {elapsed:.1f}s "
              f"({frame_count / max(elapsed, 1e-6):.1f} fps) to {save_dir}/")


# ----------------------------------------------------------------------
# Standalone usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RealSense camera capture")
    parser.add_argument("--mode", choices=["preview", "collect", "record"],
                        default="collect",
                        help="preview: live view only | collect: 2 windows + save | record: headless save")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (record mode)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (record mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (record mode)")
    parser.add_argument("--no-filters", action="store_true",
                        help="Disable depth post-processing filters")
    args = parser.parse_args()

    with RealSenseCamera(width=args.width, height=args.height, fps=args.fps,
                         enable_filters=not args.no_filters) as cam:
        if args.mode == "preview":
            cam.preview()
        elif args.mode == "collect":
            cam.collect(save_dir=args.output, max_frames=args.max_frames,
                        duration_sec=args.duration)
        else:
            cam.record(save_dir=args.output, max_frames=args.max_frames,
                       duration_sec=args.duration)
