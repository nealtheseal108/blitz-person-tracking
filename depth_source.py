"""
Depth-camera source abstraction.

The whole point of switching to a depth camera is privacy: we get
true distance per pixel without any biometric capture. The RGB stream
is only used briefly for YOLO person detection and is then immediately
discarded -- it never reaches a writer or disk.

Two backends, both auto-detected:

    RealSenseSource   – Intel RealSense D435 / D455 (pyrealsense2 SDK)
    OakDSource        – Luxonis OAK-D-Lite / OAK-D (depthai SDK)

Both expose the same interface:

    src = open_depth_source("realsense")        # or "oakd", or "auto"
    for frame in src.frames():
        depth_m = frame.depth_meters             # HxW float32, meters
        rgb = frame.rgb                          # HxW BGR uint8 (transient)
        # ... use rgb for detection, then it goes out of scope
    src.close()

`depth_meters` is float32 with NaN where the sensor reports no return
(specular surfaces, out-of-range, etc.). Always check for NaN/inf
before using a sample.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class DepthFrame:
    depth_meters: np.ndarray   # float32 HxW, meters; NaN = no return
    rgb: np.ndarray            # uint8 HxW BGR; intended to be ephemeral
    timestamp_sec: float
    frame_index: int

    def release_rgb(self) -> None:
        """Explicitly drop the RGB array. Call as soon as detection is done.

        After this call, `self.rgb` is replaced with a 1x1 black pixel so
        any accidental writer that still references the frame can't leak
        a recognizable image.
        """
        self.rgb = np.zeros((1, 1, 3), dtype=np.uint8)


class DepthSource:
    """Common interface."""
    width: int
    height: int
    fps: float

    def frames(self) -> Iterator[DepthFrame]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "DepthSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ── Intel RealSense backend ─────────────────────────────────────────────────

class RealSenseSource(DepthSource):
    """Intel RealSense D4xx series via pyrealsense2.

    pip install pyrealsense2

    Configures aligned color+depth at the requested resolution. Depth is
    converted to meters using the sensor's depth_scale.
    """

    def __init__(self, width: int = 848, height: int = 480, fps: int = 30,
                 align_to_color: bool = True):
        try:
            import pyrealsense2 as rs  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyrealsense2 not installed. Run: pip install pyrealsense2"
            ) from e
        import pyrealsense2 as rs

        self._rs = rs
        self.width = width
        self.height = height
        self.fps = float(fps)

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = self._pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()  # meters per unit

        self._align: Optional["rs.align"] = (
            rs.align(rs.stream.color) if align_to_color else None
        )

        self._frame_idx = 0

    def frames(self) -> Iterator[DepthFrame]:
        rs = self._rs
        try:
            while True:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                if self._align is not None:
                    frames = self._align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_raw = np.asanyarray(depth_frame.get_data())   # uint16
                depth_m = depth_raw.astype(np.float32) * self._depth_scale
                # Sentinel: 0 in z16 means "no return". Mark as NaN.
                depth_m[depth_raw == 0] = np.nan

                rgb = np.asanyarray(color_frame.get_data())         # BGR uint8
                ts = depth_frame.get_timestamp() / 1000.0           # ms -> s

                self._frame_idx += 1
                yield DepthFrame(
                    depth_meters=depth_m,
                    rgb=rgb,
                    timestamp_sec=ts,
                    frame_index=self._frame_idx,
                )
        except (RuntimeError, KeyboardInterrupt):
            return

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ── Luxonis OAK-D backend ───────────────────────────────────────────────────

class OakDSource(DepthSource):
    """Luxonis OAK-D-Lite / OAK-D via depthai.

    pip install depthai

    Builds a small pipeline: stereo depth (in mm) + a low-res color
    stream aligned to the depth output. We convert depth to meters and
    expose the same DepthFrame interface as the RealSense path.
    """

    def __init__(self, width: int = 640, height: int = 400, fps: int = 30):
        try:
            import depthai as dai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "depthai not installed. Run: pip install depthai"
            ) from e
        import depthai as dai

        self._dai = dai
        self.width = width
        self.height = height
        self.fps = float(fps)

        pipeline = dai.Pipeline()

        mono_left = pipeline.createMonoCamera()
        mono_right = pipeline.createMonoCamera()
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)
        # Align depth to the color camera so xy in RGB == xy in depth
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setIspScale(1, 3)   # ~640x360-ish; we don't need 1080p
        cam_rgb.setFps(fps)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)

        self._device = dai.Device(pipeline)
        self._q_depth = self._device.getOutputQueue("depth", maxSize=4, blocking=False)
        self._q_rgb = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)

        self._frame_idx = 0

    def frames(self) -> Iterator[DepthFrame]:
        import time as _time
        try:
            while True:
                d = self._q_depth.get()
                c = self._q_rgb.get()
                if d is None or c is None:
                    continue

                depth_raw = d.getFrame()                       # uint16, mm
                depth_m = depth_raw.astype(np.float32) / 1000.0
                depth_m[depth_raw == 0] = np.nan

                rgb = c.getCvFrame()                           # BGR uint8

                self._frame_idx += 1
                yield DepthFrame(
                    depth_meters=depth_m,
                    rgb=rgb,
                    timestamp_sec=_time.time(),
                    frame_index=self._frame_idx,
                )
        except (RuntimeError, KeyboardInterrupt):
            return

    def close(self) -> None:
        try:
            self._device.close()
        except Exception:
            pass


# ── Factory ─────────────────────────────────────────────────────────────────

def open_depth_source(backend: str = "auto", **kwargs) -> DepthSource:
    """Open whichever depth camera is connected.

    backend:
        "realsense"  – force RealSense
        "oakd"       – force OAK-D
        "auto"       – try RealSense first, fall back to OAK-D
    """
    backend = backend.lower()
    if backend == "realsense":
        return RealSenseSource(**kwargs)
    if backend == "oakd":
        return OakDSource(**kwargs)
    if backend == "auto":
        last_err: Optional[Exception] = None
        for cls in (RealSenseSource, OakDSource):
            try:
                return cls(**{k: v for k, v in kwargs.items()
                              if k in cls.__init__.__code__.co_varnames})
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"No depth camera found. Tried RealSense and OAK-D. Last error: {last_err}"
        )
    raise ValueError(f"unknown depth backend: {backend!r}")


@contextmanager
def depth_source(backend: str = "auto", **kwargs):
    src = open_depth_source(backend, **kwargs)
    try:
        yield src
    finally:
        src.close()


# ── Helpers for visualization (operates on DEPTH only, never RGB) ────────────

def colorize_depth(depth_m: np.ndarray, max_m: float = 6.0) -> np.ndarray:
    """Return a BGR uint8 visualization of a depth map.

    Used for the annotated output video. Because we render from depth
    rather than RGB, the saved video literally cannot show faces.
    """
    import cv2
    finite = np.where(np.isfinite(depth_m), depth_m, 0.0)
    norm = np.clip(finite / max_m, 0.0, 1.0)
    norm_u8 = (255 * (1.0 - norm)).astype(np.uint8)  # near = bright
    color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    # Mask out NaN regions to black
    color[~np.isfinite(depth_m)] = (0, 0, 0)
    return color


def sample_depth(depth_m: np.ndarray, x: int, y: int, k: int = 5) -> float:
    """Median depth in a small kxk window around (x,y), in meters.

    Median is more robust than a single-pixel sample on noisy stereo
    depth. Returns NaN if every pixel in the window is invalid.
    """
    h, w = depth_m.shape[:2]
    half = k // 2
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))
