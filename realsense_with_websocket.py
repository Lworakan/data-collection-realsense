"""
RealSense Sync Listener

Sync-only Socket.IO client for Intel RealSense recording.
It joins a session room and reacts to backend start/stop events:
- start_recording: begin local RealSense capture loop
- stop_recording: stop capture loop and save files

No database schema changes are required because this client uses
`join_sync_listener` (room subscription only, no device row writes).
"""

import argparse
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import socketio

# Assumes your RealSenseCamera class is in realsense_camera.py
# (use the class you pasted in your message).
from realsense_camera import RealSenseCamera


class RealSenseSyncListener:
    def __init__(
        self,
        ws_url: str,
        session_id: str,
        output_root: str,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        enable_filters: bool = True,
        source_name: str = "realsense-depth",
    ):
        self.ws_url = ws_url.rstrip("/")
        self.session_id = session_id
        self.output_root = output_root
        self.source_name = source_name

        self.camera = RealSenseCamera(
            width=width,
            height=height,
            fps=fps,
            enable_filters=enable_filters,
        )

        self.sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False)

        self._recording_lock = threading.Lock()
        self._recording_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_recording = False

        self._recording_dir: Optional[Path] = None
        self._current_posture: str = "unknown"
        self._current_distance: str = "nom"
        self._recording_started_at: Optional[float] = None
        self._frame_count = 0

        self._bind_socket_events()

    def _bind_socket_events(self) -> None:
        @self.sio.event
        def connect():
            print(f"[Sync] Connected to {self.ws_url}")
            self.sio.emit(
                "join_sync_listener",
                {
                    "sessionId": self.session_id,
                    "source": self.source_name,
                    "userAgent": "python-realsense-sync-listener",
                },
            )

        @self.sio.event
        def disconnect():
            print("[Sync] Disconnected from server")

        @self.sio.on("joined_sync_listener")
        def on_joined_sync_listener(data):
            print(f"[Sync] Joined session room: {data}")

        @self.sio.on("start_recording")
        def on_start_recording(data):
            posture_label = data.get("postureLabel", "unknown")
            duration_ms = int(data.get("duration", 0))
            distance = data.get("distance", "nom")
            ts = data.get("timestamp")
            print(
                f"[Sync] start_recording posture={posture_label} distance={distance} "
                f"duration_ms={duration_ms} ts={ts}"
            )
            self.start_recording(posture_label=posture_label, distance=distance)

        @self.sio.on("stop_recording")
        def on_stop_recording(data):
            ts = data.get("timestamp")
            print(f"[Sync] stop_recording ts={ts}")
            self.stop_recording()

        @self.sio.on("error")
        def on_error(data):
            print(f"[Sync] server error: {data}")

    def _build_output_dir(self, posture_label: str, distance: str) -> Path:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (
            Path(self.output_root)
            / "sessions"
            / self.session_id
            / "realsense-depth"
            / distance
            / posture_label
            / now
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _record_loop(self) -> None:
        assert self._recording_dir is not None
        print(f"[Record] Saving to {self._recording_dir}")

        self._recording_started_at = time.time()
        self._frame_count = 0

        while not self._stop_event.is_set():
            frame_data = self.camera.capture_frame()
            frame_id = f"{self._frame_count:06d}"
            self.camera.save_frame(frame_data, str(self._recording_dir), frame_id)
            self._frame_count += 1

            if self._frame_count % 30 == 0:
                elapsed = time.time() - self._recording_started_at
                fps = self._frame_count / max(elapsed, 1e-6)
                print(
                    f"[Record] posture={self._current_posture} distance={self._current_distance} "
                    f"frames={self._frame_count} elapsed={elapsed:.1f}s fps={fps:.1f}"
                )

        elapsed = time.time() - (self._recording_started_at or time.time())
        fps = self._frame_count / max(elapsed, 1e-6)
        print(
            f"[Record] Stopped. frames={self._frame_count} elapsed={elapsed:.1f}s avg_fps={fps:.1f}"
        )

    def start_recording(self, posture_label: str, distance: str) -> None:
        with self._recording_lock:
            if self._is_recording:
                print("[Record] Already recording; ignoring duplicate start signal")
                return

            if not getattr(self.camera, "_running", False):
                self.camera.start()

            self._current_posture = posture_label
            self._current_distance = distance
            self._recording_dir = self._build_output_dir(posture_label, distance)

            self._stop_event.clear()
            self._recording_thread = threading.Thread(
                target=self._record_loop,
                daemon=True,
                name="realsense-record-thread",
            )
            self._is_recording = True
            self._recording_thread.start()
            print("[Record] Started")

    def stop_recording(self) -> None:
        with self._recording_lock:
            if not self._is_recording:
                print("[Record] Not recording; ignoring stop signal")
                return

            self._stop_event.set()
            if self._recording_thread and self._recording_thread.is_alive():
                self._recording_thread.join(timeout=10)

            self._is_recording = False
            self._recording_thread = None
            print("[Record] Stop complete")

    def run_forever(self) -> None:
        Path(self.output_root).mkdir(parents=True, exist_ok=True)

        print("[Sync] Starting RealSense sync listener")
        print(f"[Sync] session_id={self.session_id}")
        print(f"[Sync] output_root={self.output_root}")

        self.sio.connect(self.ws_url, transports=["websocket"])

        try:
            self.sio.wait()
        except KeyboardInterrupt:
            print("\n[Sync] Keyboard interrupt received")
        finally:
            self.stop_recording()
            try:
                self.camera.stop()
            except Exception as exc:
                print(f"[Sync] camera stop warning: {exc}")
            if self.sio.connected:
                self.sio.disconnect()
            print("[Sync] Exited")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RealSense sync listener for session start/stop recording"
    )
    parser.add_argument("--ws-url", default="http://localhost:3001", help="Socket.IO server URL")
    parser.add_argument("--session-id", required=True, help="Session UUID to subscribe to")
    parser.add_argument(
        "--output-root",
        default=os.path.join("datasets", "realsense_sync"),
        help="Root output directory",
    )
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-filters", action="store_true", help="Disable depth filters")
    parser.add_argument(
        "--source-name",
        default="realsense-depth",
        help="Listener source label for logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    listener = RealSenseSyncListener(
        ws_url=args.ws_url,
        session_id=args.session_id,
        output_root=args.output_root,
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_filters=not args.no_filters,
        source_name=args.source_name,
    )
    listener.run_forever()


if __name__ == "__main__":
    main()

