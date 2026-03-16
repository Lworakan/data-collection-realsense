# data-collection-realsense

Intel RealSense D400 series (D415/D435/D435i) camera capture module for the data collection pipeline. Captures synchronized color frames, colorized depth frames, and raw depth arrays.

## Setup

### 1. Install system driver (Linux)

```bash
sudo apt-get install librealsense2-dkms librealsense2-utils
```

> For other OS see [librealsense installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)

### 2. Install Python dependencies

```bash
pip install pyrealsense2 numpy opencv-python
```

### 3. Verify camera connection

```bash
# Should list your RealSense device
rs-enumerate-devices | head -20
```

## Usage

### Collect data (2 windows + save)

Opens two live windows (color and depth) and saves every frame to disk.

```bash
python realsense_camera.py --mode collect
```

Options:

```bash
# Collect 100 frames
python realsense_camera.py --mode collect --max-frames 100

# Collect for 30 seconds
python realsense_camera.py --mode collect --duration 30

# Custom output directory
python realsense_camera.py --mode collect --output datasets/realsense/my_session

# Custom resolution
python realsense_camera.py --mode collect --width 640 --height 480 --fps 30
```

Press `q` to stop, `p` to pause/resume saving.

### Preview only (no save)

```bash
python realsense_camera.py --mode preview
```

### Headless record (no GUI)

```bash
python realsense_camera.py --mode record --max-frames 500
```

### All CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `collect` | `preview` / `collect` / `record` |
| `--width` | `848` | Stream width (px) |
| `--height` | `480` | Stream height (px) |
| `--fps` | `30` | Framerate |
| `--max-frames` | None | Stop after N frames |
| `--duration` | None | Stop after N seconds |
| `--output` | auto | Output directory |
| `--no-filters` | False | Disable depth post-processing |

## Use as a Python class

```python
from realsense_camera import RealSenseCamera

cam = RealSenseCamera(width=848, height=480, fps=30)
cam.start()

# Capture a frame
frame = cam.capture_frame()
print(frame["color_image"].shape)   # (480, 848, 3)
print(frame["depth_raw"].shape)     # (480, 848)
print(frame["depth_raw"].dtype)     # float32

# Save to disk
cam.save_frame(frame, "datasets/realsense/session_01", "000000")

cam.stop()
```

Or with context manager:

```python
with RealSenseCamera() as cam:
    for i in range(100):
        frame = cam.capture_frame()
        cam.save_frame(frame, "output/session", f"{i:06d}")
```

## Integrating with data_collection app

```python
from realsense_camera import RealSenseCamera

class DataCollectionApp:
    def __init__(self):
        # ... existing phone_cam, notebook_cam ...
        self.realsense = RealSenseCamera()

    def start_all(self):
        self.realsense.start()

    def capture_all(self):
        rs_frame = self.realsense.capture_frame()
        color = rs_frame["color_image"]       # (480, 848, 3) uint8 BGR
        depth_vis = rs_frame["depth_image"]   # (480, 848, 3) uint8 BGR
        depth_m = rs_frame["depth_raw"]       # (480, 848) float32 meters
        self.realsense.save_frame(rs_frame, "datasets/realsense/session", "000000")

    def stop_all(self):
        self.realsense.stop()
```

## Output folder structure

```
datasets/realsense/
└── 20260316_143022/         # auto-generated session ID
    ├── color/               # BGR color images
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── depth/               # colorized depth visualization
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── depth_raw/           # raw depth in meters
        ├── 000000.npy
        ├── 000001.npy
        └── ...
```

## Depth data

| Property | Value |
|----------|-------|
| Unit | Meters (float32) |
| Alignment | Depth aligned to color (pixel-to-pixel) |
| Invalid pixels | `0.0` = no depth reading |
| Range | ~0.1m - 10m (D435), ~0.16m - 10m (D415) |
| Filters | Spatial, Temporal, Hole-filling (disable with `--no-filters`) |

### Loading saved depth

```python
import numpy as np
depth = np.load("datasets/realsense/session/depth_raw/000000.npy")
# depth.shape = (480, 848), dtype = float32, values in meters
```
