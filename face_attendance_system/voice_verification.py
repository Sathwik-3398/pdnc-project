"""
voice_verification.py
---------------------
Lightweight voice verification module.

Records audio via sounddevice (already installed by mediapipe),
then uses Google Speech Recognition (free, no API key) to transcribe.

Entry point:
    result = verify_voice(cap, draw_hud_fn, draw_box_fn, ...)
    # result → "MATCH" | "NO_MATCH" | "ERROR"
"""

import threading
import wave
import tempfile
import os
import sys
import time
import cv2
import numpy as np

# ── Try loading audio / SR libs ──────────────────────────────────────────────
try:
    import sounddevice as sd
    _SD_OK = True
except Exception as e:
    _SD_OK = False
    print(f"[VOICE] sounddevice unavailable: {e}", file=sys.stderr)

try:
    import speech_recognition as sr
    _SR_OK = True
except ImportError:
    _SR_OK = False
    print("[VOICE] speech_recognition not installed. Run: pip install SpeechRecognition", file=sys.stderr)

VOICE_AVAILABLE = _SD_OK and _SR_OK

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
RECORD_SECONDS   = 4
EXPECTED_PHRASE  = "present"      # must contain this word (case-insensitive)


def _record_audio():
    """Record RECORD_SECONDS of mono 16-bit PCM. Returns numpy array."""
    data = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    return data


def _save_wav(audio_data, path):
    """Write numpy int16 array to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())


def _transcribe(wav_path):
    """Transcribe a WAV file using Google Web Speech API. Returns text or ''."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        print(f"[VOICE] Heard: '{text}'", file=sys.stderr)
        return text.lower()
    except sr.UnknownValueError:
        print("[VOICE] Could not understand audio.", file=sys.stderr)
        return ""
    except sr.RequestError as e:
        print(f"[VOICE] Google SR error: {e}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"[VOICE] Transcription error: {e}", file=sys.stderr)
        return ""


def verify_voice(cap, draw_hud_fn, draw_box_fn,
                 face_top, face_right, face_bottom, face_left,
                 recognized_name):
    """
    Full voice-verification flow with live camera feed during recording.

    Parameters
    ----------
    cap           : open cv2.VideoCapture
    draw_hud_fn   : draw_hud(frame, text, color)
    draw_box_fn   : draw_targeting_box(frame, top, right, bottom, left, ...)
    face_*        : bounding box of the recognised face

    Returns
    -------
    "MATCH"    – phrase detected → mark attendance
    "NO_MATCH" – recording succeeded but phrase not heard
    "ERROR"    – audio/SR failure (treat as absent)
    """
    if not VOICE_AVAILABLE:
        print("[VOICE] Libraries unavailable — skipping voice check.", file=sys.stderr)
        return "MATCH"   # graceful degradation

    # ── Phase 1: countdown "Get ready to speak" (2 s) ──────────────────────
    countdown_start = time.time()
    while time.time() - countdown_start < 2.0:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        remaining = max(0, 2.0 - (time.time() - countdown_start))
        draw_hud_fn(frame, f"Say  'Present Sir'  in {remaining:.1f}s...", (0, 200, 255))
        draw_box_fn(frame, face_top, face_right, face_bottom, face_left,
                    color=(0, 200, 255), thickness=2)
        h = frame.shape[0]
        cv2.putText(frame,
                    f"{recognized_name.upper()}  |  Voice verification",
                    (30, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Face Attendance Scanner", frame)
        cv2.waitKey(1)

    # ── Phase 2: record audio in background; show "Listening..." on camera ──
    audio_result  = [None]
    record_done   = [False]
    record_error  = [False]

    def _record_thread():
        try:
            audio_result[0] = _record_audio()
        except Exception as e:
            print(f"[VOICE] Record error: {e}", file=sys.stderr)
            record_error[0] = True
        finally:
            record_done[0] = True

    t = threading.Thread(target=_record_thread, daemon=True)
    t.start()

    listen_start = time.time()
    while not record_done[0]:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        elapsed = time.time() - listen_start
        bar_len  = int((elapsed / RECORD_SECONDS) * 200)
        dot_anim = "." * (int(elapsed * 3) % 4)

        draw_hud_fn(frame, f"Listening{dot_anim}  ({RECORD_SECONDS - elapsed:.1f}s)", (0, 80, 255))
        draw_box_fn(frame, face_top, face_right, face_bottom, face_left,
                    color=(0, 80, 255), thickness=2)

        # Progress bar
        h, w = frame.shape[:2]
        bar_x, bar_y, bar_h = 30, h - 45, 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + bar_h), (0, 80, 255), -1)
        cv2.putText(frame, "SAY: 'PRESENT SIR'",
                    (30, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)

        cv2.imshow("Face Attendance Scanner", frame)
        cv2.waitKey(1)

    t.join()

    if record_error[0] or audio_result[0] is None:
        return "ERROR"

    # ── Phase 3: transcribe ──────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        _save_wav(audio_result[0], tmp.name)
        text = _transcribe(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    # ── Phase 4: match check ─────────────────────────────────────────────────
    matched = EXPECTED_PHRASE in text
    return "MATCH" if matched else "NO_MATCH"
