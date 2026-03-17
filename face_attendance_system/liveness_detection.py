"""
liveness_detection.py
---------------------
Anti-Spoofing / Liveness Detection via Eye Blink (EAR).
Uses MediaPipe Tasks FaceLandmarker API (mediapipe >= 0.10.13).

Logic:
  - Compute Eye Aspect Ratio (EAR) from 6-point eye landmarks.
  - Blink = EAR drops below computed adaptive threshold.
  - If at least 1 blink detected within TIMEOUT → LIVE (True).
  - Else → SPOOF / no cooperation (False).
"""

import cv2
import numpy as np
import time
import os
import sys

# ── MediaPipe Tasks import ───────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except Exception as e:
    _MP_AVAILABLE = False
    print(f"[WARNING] mediapipe import failed: {e}", file=sys.stderr)

# ── Model path ───────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_landmarker.task"
)

# ── Eye landmark indices (MediaPipe 478-point mesh) ──────────────────────────
# These 6-point eye contours give the classic EAR formula
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def _ear(landmarks, indices):
    """Eye Aspect Ratio from 6 (x,y) points."""
    pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.3


class LivenessDetector:
    """
    Blink-based liveness check using MediaPipe FaceLandmarker (Tasks API).

    Tunable class constants
    -----------------------
    CALIBRATION_FRAMES  : frames used to compute open-eye baseline EAR
    THRESHOLD_RATIO     : blink threshold = open_ear * THRESHOLD_RATIO
    MIN_THRESHOLD       : floor so very loose faces still trigger
    BLINK_CONSEC_FRAMES : consecutive sub-threshold frames = 1 blink
    BLINKS_REQUIRED     : blinks needed to pass
    TIMEOUT_SECONDS     : window given to the user
    """

    CALIBRATION_FRAMES  = 20    # ~0.7 s of calibration at 30 fps
    THRESHOLD_RATIO     = 0.75  # blink when EAR < open_ear * 0.75
    MIN_THRESHOLD       = 0.18  # never go above this even if ratio gives more
    BLINK_CONSEC_FRAMES = 2
    BLINKS_REQUIRED     = 1
    TIMEOUT_SECONDS     = 7.0

    def __init__(self):
        self.available   = _MP_AVAILABLE and os.path.exists(_MODEL_PATH)
        self._landmarker = None

        if _MP_AVAILABLE and not os.path.exists(_MODEL_PATH):
            print(f"[WARNING] face_landmarker.task not found at {_MODEL_PATH}", file=sys.stderr)

    # ── Internal: lazy-init FaceLandmarker ──────────────────────────────────

    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        base_opts = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,   # synchronous per-frame
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        return self._landmarker

    # ── Internal: compute EAR from one frame ────────────────────────────────

    def _get_ear(self, frame):
        """
        Returns (ear_value, face_detected).
        ear_value is -1.0 when no face found.
        """
        try:
            landmarker = self._get_landmarker()
            h, w = frame.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                return -1.0, False

            lms      = result.face_landmarks[0]
            left_ear  = _ear(lms, LEFT_EYE)
            right_ear = _ear(lms, RIGHT_EYE)
            return (left_ear + right_ear) / 2.0, True

        except Exception as e:
            print(f"[EAR ERROR] {e}", file=sys.stderr)
            return -1.0, False

    # ── Public API ───────────────────────────────────────────────────────────

    def run_blink_check(self, cap, draw_hud_fn, draw_box_fn,
                        face_top, face_right, face_bottom, face_left,
                        recognized_name):
        """
        Runs blink-check then voice-check on the already-open `cap`.

        Returns
        -------
        'LIVE'       – blink + voice both verified  → mark attendance
        'SPOOF'      – blink not detected within timeout
        'VOICE_FAIL' – blink verified but did NOT say 'Present Sir'
        'ERROR'      – unexpected failure
        """
        if not self.available:
            print("[LIVENESS] MediaPipe not available, skipping check.", file=sys.stderr)
            return "LIVE"   # graceful degradation

        try:
            return self._blink_loop(
                cap, draw_hud_fn, draw_box_fn,
                face_top, face_right, face_bottom, face_left,
                recognized_name
            )
        except Exception as e:
            print(f"[LIVENESS CRITICAL] {type(e).__name__}: {e}", file=sys.stderr)
            return "ERROR"

    def _blink_loop(self, cap, draw_hud_fn, draw_box_fn,
                    face_top, face_right, face_bottom, face_left,
                    recognized_name):

        # ── 1. Flush stale buffer ────────────────────────────────────────────
        for _ in range(5):
            cap.read()

        # ── 2. Calibration: measure open-eye EAR baseline ───────────────────
        open_ears    = []
        calib_start  = time.time()
        calib_failed = False

        while len(open_ears) < self.CALIBRATION_FRAMES:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Show "Keep eyes open" message during calibration
            h_f = frame.shape[0]
            draw_hud_fn(frame, "Hold still — calibrating...", (0, 200, 255))
            draw_box_fn(frame, face_top, face_right, face_bottom, face_left,
                        color=(0, 200, 255), thickness=2)
            pct = int(len(open_ears) / self.CALIBRATION_FRAMES * 100)
            cv2.putText(frame, f"Calibrating: {pct}%  (keep eyes OPEN)",
                        (30, h_f - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1)
            cv2.imshow("Face Attendance Scanner", frame)
            cv2.waitKey(1)

            ear_val, found = self._get_ear(frame)
            if found and ear_val > 0.15:   # sane open-eye range
                open_ears.append(ear_val)

            if time.time() - calib_start > 3.0:
                calib_failed = True
                break

        if calib_failed or len(open_ears) < 5:
            # Could not calibrate — use a safe fixed threshold
            threshold = self.MIN_THRESHOLD
            print("[LIVENESS] Calibration failed, using fixed threshold.", file=sys.stderr)
        else:
            open_avg  = np.mean(open_ears)
            threshold = min(open_avg * self.THRESHOLD_RATIO, self.MIN_THRESHOLD)
            print(f"[LIVENESS] Baseline EAR={open_avg:.3f}, threshold={threshold:.3f}", file=sys.stderr)

        # ── 3. Blink detection loop ──────────────────────────────────────────
        blink_ctr    = 0
        total_blinks = 0
        read_fails   = 0
        face_covered_streak = 0  # NEW: track obscured face
        start_time   = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                read_fails += 1
                if read_fails >= 15:
                    return "SPOOF"
                time.sleep(0.03)
                continue
            read_fails = 0

            elapsed   = time.time() - start_time
            remaining = max(0.0, self.TIMEOUT_SECONDS - elapsed)

            if elapsed > self.TIMEOUT_SECONDS:
                self._draw_spoof_screen(frame, draw_hud_fn, draw_box_fn,
                                        face_top, face_right, face_bottom, face_left)
                cv2.imshow("Face Attendance Scanner", frame)
                cv2.waitKey(2000)
                return "SPOOF"

            ear_val, face_found = self._get_ear(frame)

            # Update blink counter only when face is tracked
            if face_found and ear_val >= 0:
                face_covered_streak = 0
                if ear_val < threshold:
                    blink_ctr += 1
                else:
                    if blink_ctr >= self.BLINK_CONSEC_FRAMES:
                        total_blinks += 1
                        print(f"[LIVENESS] Blink #{total_blinks} detected! EAR={ear_val:.3f}", file=sys.stderr)
                    blink_ctr = 0
            else:
                face_covered_streak += 1

            if total_blinks >= self.BLINKS_REQUIRED:
                # ── Blink verified → now check voice ────────────────────────
                from voice_verification import verify_voice
                voice_result = verify_voice(
                    cap, draw_hud_fn, draw_box_fn,
                    face_top, face_right, face_bottom, face_left,
                    recognized_name
                )

                if voice_result == "MATCH":
                    # Both blink + voice passed
                    ret2, frame2 = cap.read()
                    if not ret2 or frame2 is None:
                        frame2 = frame
                    self._draw_verified_screen(frame2, draw_hud_fn, draw_box_fn,
                                               face_top, face_right, face_bottom, face_left,
                                               recognized_name)
                    cv2.imshow("Face Attendance Scanner", frame2)
                    cv2.waitKey(1500)
                    return "LIVE"
                else:
                    # Voice failed
                    ret2, frame2 = cap.read()
                    if not ret2 or frame2 is None:
                        frame2 = frame
                    self._draw_voice_failed_screen(frame2, draw_hud_fn, draw_box_fn,
                                                   face_top, face_right, face_bottom, face_left)
                    cv2.imshow("Face Attendance Scanner", frame2)
                    cv2.waitKey(2000)
                    return "VOICE_FAIL"

            # Draw prompt
            is_covered = (face_covered_streak > 15)
            self._draw_blink_prompt(frame, draw_hud_fn, draw_box_fn,
                                    face_top, face_right, face_bottom, face_left,
                                    recognized_name, remaining,
                                    ear_val, threshold, face_found, is_covered)
            cv2.imshow("Face Attendance Scanner", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return "SPOOF"

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def _draw_blink_prompt(self, frame, draw_hud_fn, draw_box_fn,
                           top, right, bottom, left,
                           name, remaining, ear_val, threshold, face_found, is_covered):
        pulse = int((np.sin(time.time() * 4) + 1) / 2 * 80)
        color = (0, 0, 255) if is_covered else (0, 200 + pulse, 200 + pulse)
        
        status_msg = "UNCOVER YOUR FACE!" if is_covered else f"Blink to Verify  [{remaining:.1f}s]"
        draw_hud_fn(frame, status_msg, color)
        draw_box_fn(frame, top, right, bottom, left, color=color, thickness=2)

        h = frame.shape[0]
        if face_found and ear_val >= 0:
            is_blinking = ear_val < threshold
            ear_color = (0, 60, 255) if is_blinking else (180, 180, 180)
            sub = f"{name.upper()}  |  EAR: {ear_val:.3f}  thresh: {threshold:.3f}  {'<< BLINK' if is_blinking else ''}"
        else:
            ear_color = (40, 100, 255)
            sub = f"{name.upper()}  |  Face not tracked - Please look at camera"

        cv2.putText(frame, sub, (30, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, ear_color, 1)

    def _draw_verified_screen(self, frame, draw_hud_fn, draw_box_fn,
                              top, right, bottom, left, name):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        draw_box_fn(frame, top, right, bottom, left, color=(0, 255, 0), thickness=3, draw_solid=True)
        draw_hud_fn(frame, f"EYE + VOICE VERIFIED — {name.upper()}", (0, 255, 0))

    def _draw_spoof_screen(self, frame, draw_hud_fn, draw_box_fn, top, right, bottom, left):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        draw_box_fn(frame, top, right, bottom, left, color=(0, 0, 255), thickness=3, draw_solid=True)
        draw_hud_fn(frame, "SPOOF DETECTED - Blink Not Verified", (0, 0, 255))

    def _draw_voice_failed_screen(self, frame, draw_hud_fn, draw_box_fn, top, right, bottom, left):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 60, 160), -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        draw_box_fn(frame, top, right, bottom, left, color=(0, 100, 255), thickness=3, draw_solid=True)
        draw_hud_fn(frame, "VOICE FAILED - 'Present Sir' Not Heard", (0, 100, 255))
