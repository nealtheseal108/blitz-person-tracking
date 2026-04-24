"""
Head + body detector — two MediaPipe layers, no Haar, no YOLO.

Layer 1 — FaceLandmarker (frontal/near-profile faces)
  Gives precise yaw → classifies LOOKING (frontal) vs PASSING (profile).

Layer 2 — PoseLandmarker (whole-body pose, works at ALL angles)
  Constructs a head+torso bbox from visible keypoints.
  Any pose hit NOT already covered by a face detection → "profile" / PASSING.
  This catches people walking past side-on where the face layer fires nothing.

Both layers use the MediaPipe Tasks API (mediapipe >= 0.10.14).
Models are downloaded automatically (~2 MB face, ~4 MB pose) on first run.

Orientation:
    frontal  |yaw| < 30 deg  -> LOOKING at machine
    profile  |yaw| < 75 deg  -> PASSING (side view, or pose-only detection)
    away     beyond / undetected -> filtered out
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Yaw thresholds ────────────────────────────────────────────────────────────
LOOK_YAW_DEG = 40.0
SIDE_YAW_DEG = 75.0

# ── Model files ───────────────────────────────────────────────────────────────
_FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_POSE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_DIR         = os.path.dirname(os.path.abspath(__file__))
_FACE_MODEL  = os.path.join(_DIR, "face_landmarker.task")
_POSE_MODEL  = os.path.join(_DIR, "pose_landmarker_lite.task")


def _ensure(url: str, path: str) -> str:
    if not os.path.exists(path):
        name = os.path.basename(path)
        print(f"[face_detector] Downloading {name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[face_detector] Saved {name}")
    return path


# ── IoU / NMS ─────────────────────────────────────────────────────────────────

def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: List["FaceDetection"], iou_thresh: float = 0.35) -> List["FaceDetection"]:
    """Non-maximum suppression — merge duplicate detections of the same face.

    Keeps the detection with the largest area when two overlap significantly.
    Prevents FaceLandmarker from double-detecting a large close-up face.
    """
    if len(dets) <= 1:
        return dets
    # Sort largest area first — largest bbox is usually the most complete detection
    by_area = sorted(dets, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
    kept: List["FaceDetection"] = []
    for d in by_area:
        if not any(_iou(d.x1, d.y1, d.x2, d.y2,
                        k.x1, k.y1, k.x2, k.y2) > iou_thresh
                   for k in kept):
            kept.append(d)
    return kept


# ── Detection dataclass ───────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """One detected head/person with bbox and orientation."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    yaw_deg:     float = 0.0
    orientation: str   = "frontal"   # "frontal" | "profile"
    appearance_hist: Optional[np.ndarray] = None   # duck-type for CentroidTracker

    @property
    def centroid(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def foot_point(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)

    @property
    def is_looking(self) -> bool:
        return self.orientation == "frontal"

    @property
    def is_visible(self) -> bool:
        return self.orientation in ("frontal", "profile")

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)


# ── Landmark helpers ──────────────────────────────────────────────────────────

def _yaw_from_landmarks(landmarks, img_w: int, img_h: int) -> float:
    nose_x  = landmarks[1].x
    left_x  = landmarks[33].x
    right_x = landmarks[263].x
    eye_mid  = 0.5 * (left_x + right_x)
    eye_half = max(1e-6, abs(right_x - left_x) * 0.5)
    return float(np.clip((nose_x - eye_mid) / eye_half * 35.0, -75.0, 75.0))


def _classify(yaw: float) -> str:
    a = abs(yaw)
    if a <= LOOK_YAW_DEG: return "frontal"
    if a <= SIDE_YAW_DEG: return "profile"
    return "away"


def _lms_to_face_det(lms, w: int, h: int) -> Optional[FaceDetection]:
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    px = max(6, int((max(xs) - min(xs)) * 0.10))
    py = max(6, int((max(ys) - min(ys)) * 0.12))
    x1 = int(max(0,     min(xs) - px))
    y1 = int(max(0,     min(ys) - py))
    x2 = int(min(w - 1, max(xs) + px))
    y2 = int(min(h - 1, max(ys) + py))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    yaw    = _yaw_from_landmarks(lms, w, h)
    orient = _classify(yaw)
    if orient == "away":
        return None
    return FaceDetection(x1=x1, y1=y1, x2=x2, y2=y2,
                         confidence=1.0, yaw_deg=yaw, orientation=orient)


# Pose landmark indices used for bounding-box construction
# Head: nose(0), eyes(1-6), ears(7-8)
# Torso: shoulders(11-12), hips(23-24)
_HEAD_LMS  = [0, 1, 2, 3, 4, 5, 6, 7, 8]
_TORSO_LMS = [11, 12, 23, 24]
_KEY_LMS   = _HEAD_LMS + _TORSO_LMS


def _pose_lms_to_det(lms, w: int, h: int,
                     vis_thresh: float = 0.4) -> Optional[FaceDetection]:
    """Build a head+torso FaceDetection from pose landmarks.

    Only uses keypoints whose visibility score exceeds vis_thresh so that
    out-of-frame limbs don't distort the bbox.  Returns None if too few
    keypoints are visible.
    """
    pts = [
        (lms[i].x * w, lms[i].y * h)
        for i in _KEY_LMS
        if i < len(lms) and lms[i].visibility >= vis_thresh
    ]
    if len(pts) < 3:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # Add generous padding — the pose bbox is tight to keypoints, not the
    # silhouette, so we expand it to cover the full head+upper body width.
    pad_x = max(20, int((max(xs) - min(xs)) * 0.25))
    pad_y = max(20, int((max(ys) - min(ys)) * 0.15))

    x1 = int(max(0,     min(xs) - pad_x))
    y1 = int(max(0,     min(ys) - pad_y))
    x2 = int(min(w - 1, max(xs) + pad_x))
    y2 = int(min(h - 1, max(ys) + pad_y))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    return FaceDetection(x1=x1, y1=y1, x2=x2, y2=y2,
                         confidence=0.8, yaw_deg=65.0, orientation="profile")


# ── Detector ──────────────────────────────────────────────────────────────────

class FaceDetector:
    """Two-layer head/body detector using only MediaPipe (no YOLO, no Haar).

    Layer 1 — FaceLandmarker: precise yaw, LOOKING vs PASSING for frontal faces.
    Layer 2 — PoseLandmarker: torso/body at any angle, catches side-profile
              pass-bys that FaceLandmarker misses entirely.

    Args:
        max_faces:    Max simultaneous face tracks (FaceLandmarker).
        max_people:   Max simultaneous pose tracks (PoseLandmarker).
        min_confidence: Detection threshold for both layers (0-1).
        use_pose:     Enable pose supplement (default True). Set False to
                      disable if pose causes false positives indoors.
    """

    def __init__(self, max_faces: int = 8, max_people: int = 4,
                 min_confidence: float = 0.35, use_pose: bool = True):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("pip install mediapipe") from exc

        self._mp        = mp
        self._use_tasks = False
        self._t0        = time.monotonic()
        self._prev_gray: Optional[np.ndarray] = None   # for pose motion gate

        # ── Face layer ────────────────────────────────────────────────────────
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=max_faces,
                refine_landmarks=False,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=0.5,
            )
            self._use_tasks = False
            print("[face_detector] Face: mp.solutions.face_mesh")
        else:
            from mediapipe.tasks import python as _t
            from mediapipe.tasks.python import vision as _v
            face_model = _ensure(_FACE_MODEL_URL, _FACE_MODEL)
            opts = _v.FaceLandmarkerOptions(
                base_options=_t.BaseOptions(model_asset_path=face_model),
                running_mode=_v.RunningMode.VIDEO,
                num_faces=max_faces,
                min_face_detection_confidence=min_confidence,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = _v.FaceLandmarker.create_from_options(opts)
            self._use_tasks  = True
            print("[face_detector] Face: FaceLandmarker Tasks API")

        # ── Pose layer ────────────────────────────────────────────────────────
        self._use_pose = use_pose
        if use_pose:
            self._init_pose(max_people, min_confidence)

    def _init_pose(self, max_people: int, min_confidence: float) -> None:
        mp = self._mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            # Legacy API
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=0.5,
            )
            self._pose_tasks = False
            print("[face_detector] Pose: mp.solutions.pose (legacy)")
        else:
            try:
                from mediapipe.tasks import python as _t
                from mediapipe.tasks.python import vision as _v
                pose_model = _ensure(_POSE_MODEL_URL, _POSE_MODEL)
                opts = _v.PoseLandmarkerOptions(
                    base_options=_t.BaseOptions(model_asset_path=pose_model),
                    running_mode=_v.RunningMode.VIDEO,
                    num_poses=max_people,
                    min_pose_detection_confidence=min_confidence,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose_lm = _v.PoseLandmarker.create_from_options(opts)
                self._pose_tasks = True
                print("[face_detector] Pose: PoseLandmarker Tasks API")
            except Exception as exc:
                print(f"[face_detector] Pose init failed ({exc}) — pose supplement disabled")
                self._use_pose = False

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(self, bgr_frame: np.ndarray) -> List[FaceDetection]:
        """Detect all visible heads / bodies. Returns FaceDetection list."""
        # Face layer — NMS to prevent same face being detected twice
        face_dets = _nms(self._run_face(bgr_frame), iou_thresh=0.35)

        if self._use_pose:
            curr_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            pose_dets = self._run_pose_supplement(bgr_frame, face_dets, curr_gray)
            self._prev_gray = curr_gray
        else:
            pose_dets = []

        return face_dets + pose_dets

    def close(self) -> None:
        if self._use_tasks:
            self._landmarker.close()
        else:
            self._mesh.close()
        if self._use_pose:
            if self._pose_tasks:
                self._pose_lm.close()
            else:
                self._pose.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Face layer internals ──────────────────────────────────────────────────

    def _run_face(self, bgr: np.ndarray) -> List[FaceDetection]:
        if self._use_tasks:
            return self._face_tasks(bgr)
        return self._face_solutions(bgr)

    def _face_solutions(self, bgr: np.ndarray) -> List[FaceDetection]:
        h, w = bgr.shape[:2]
        result = self._mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return []
        return [d for lm in result.multi_face_landmarks
                for d in [_lms_to_face_det(lm.landmark, w, h)] if d]

    def _face_tasks(self, bgr: np.ndarray) -> List[FaceDetection]:
        h, w = bgr.shape[:2]
        img    = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts_ms  = int((time.monotonic() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(img, ts_ms)
        if not result.face_landmarks:
            return []
        return [d for lms in result.face_landmarks
                for d in [_lms_to_face_det(lms, w, h)] if d]

    # ── Pose layer internals ──────────────────────────────────────────────────

    # ── Motion gate ───────────────────────────────────────────────────────────

    def _region_has_motion(self, curr_gray: np.ndarray,
                           x1: int, y1: int, x2: int, y2: int,
                           thresh: float = 18.0) -> bool:
        """Return True if mean pixel change in the region exceeds thresh.

        Compares current grayscale frame against the previous one.
        Static furniture / background → near-zero diff → False.
        A walking person → significant diff → True.

        thresh=18 (out of 255) is ~7% change — low enough to catch slow walkers,
        high enough to reject background noise / camera auto-exposure flicker.
        """
        if self._prev_gray is None:
            return True   # no reference yet — let first frame through
        h, w = curr_gray.shape
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return False
        diff = cv2.absdiff(self._prev_gray[y1c:y2c, x1c:x2c],
                           curr_gray[y1c:y2c, x1c:x2c])
        return float(np.mean(diff)) >= thresh

    # ── Pose supplement ───────────────────────────────────────────────────────

    def _run_pose_supplement(
        self,
        bgr: np.ndarray,
        face_dets: List[FaceDetection],
        curr_gray: np.ndarray,
    ) -> List[FaceDetection]:
        """Run pose; return only moving detections not covered by face layer."""
        pose_raw = (self._pose_tasks_detect(bgr) if self._pose_tasks
                    else self._pose_solutions_detect(bgr))

        h, w = bgr.shape[:2]
        min_side = max(60, h // 8)   # pose bbox must be at least 1/8 frame height

        # Compute minimum separation distance from any face centroid.
        # A vending machine typically has one person at a time, so any pose
        # detection near a face-tracked person is just their body — suppress it.
        # Only allow pose detections whose centroid is > half the frame height
        # away from every known face. That distance means a genuinely different
        # person further away in the scene.
        face_centroids = [((f.x1 + f.x2) // 2, (f.y1 + f.y2) // 2)
                          for f in face_dets]
        min_sep_sq = (h // 2) ** 2

        novel: List[FaceDetection] = []
        for p in pose_raw:
            # Minimum size gate
            if (p.x2 - p.x1) < min_side or (p.y2 - p.y1) < min_side:
                continue
            # Motion gate
            if not self._region_has_motion(curr_gray, p.x1, p.y1, p.x2, p.y2,
                                           thresh=25.0):
                continue
            # Spatial separation from all face-tracked people
            pcx = (p.x1 + p.x2) // 2
            pcy = (p.y1 + p.y2) // 2
            if any((pcx - fcx) ** 2 + (pcy - fcy) ** 2 < min_sep_sq
                   for fcx, fcy in face_centroids):
                continue
            # Deduplicate within pose results
            if any(_iou(p.x1, p.y1, p.x2, p.y2,
                        n.x1, n.y1, n.x2, n.y2) > 0.4
                   for n in novel):
                continue
            novel.append(p)

        return novel

    def _pose_solutions_detect(self, bgr: np.ndarray) -> List[FaceDetection]:
        h, w = bgr.shape[:2]
        result = self._pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return []
        det = _pose_lms_to_det(result.pose_landmarks.landmark, w, h)
        return [det] if det else []

    def _pose_tasks_detect(self, bgr: np.ndarray) -> List[FaceDetection]:
        h, w = bgr.shape[:2]
        img    = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts_ms  = int((time.monotonic() - self._t0) * 1000)
        result = self._pose_lm.detect_for_video(img, ts_ms)
        if not result.pose_landmarks:
            return []
        out = []
        for lms in result.pose_landmarks:
            det = _pose_lms_to_det(lms, w, h)
            if det:
                out.append(det)
        return out
