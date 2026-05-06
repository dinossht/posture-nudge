#!/usr/bin/env python3
"""Posture Nudge — single-file Tkinter app.

Designed to be packaged as a single .exe via PyInstaller. Bundles:
  - Periodic webcam posture check (MediaPipe Pose)
  - Calibration UI
  - Live test with slack slider
  - Stats tab with embedded matplotlib heatmap + daily bars
  - Slide-up chair-character popup when slouching
  - Optional autostart on Windows login

Cross-platform Python; tested on Windows 10/11 and Linux. Data lives in
the user's AppData (Windows) or ~/.local/share/posture-nudge (Linux).
"""

from __future__ import annotations

import json
import os
import platform
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

IS_WINDOWS = platform.system() == "Windows"
IS_FROZEN = getattr(sys, "frozen", False)
ASSETS_DIR = (
    Path(sys._MEIPASS) if IS_FROZEN and hasattr(sys, "_MEIPASS")  # type: ignore[attr-defined]
    else Path(__file__).resolve().parent
)
CHAIR_PNG = ASSETS_DIR / "chair.png"


def app_data_dir() -> Path:
    if IS_WINDOWS:
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData/Roaming")
        return Path(appdata) / "PostureNudge"
    return Path.home() / ".local/share/posture-nudge"


DATA_DIR = app_data_dir()
BASELINE_PATH = DATA_DIR / "baseline.json"
THRESHOLDS_PATH = DATA_DIR / "thresholds.json"
NUDGE_LOG = DATA_DIR / "nudge_log.jsonl"
CHECK_LOG = DATA_DIR / "check_log.jsonl"
SETTINGS_PATH = DATA_DIR / "settings.json"

# MediaPipe pose landmarks
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12

BG = "#0a0a0a"
FG = "#eceff4"
MUTED = "#4c566a"
ACCENT = "#bf616a"
GREEN = "#a3be8c"
YELLOW = "#ebcb8b"
RED = "#bf616a"
BLUE = "#88c0d0"


@dataclass
class Metrics:
    ear_drop: float
    shoulder_width: float
    shoulder_tilt: float
    confidence: float


@dataclass
class Thresholds:
    ear_drop: float = 0.20
    shoulder_width: float = 0.15
    shoulder_tilt: float = 0.08
    slack: int = 0

    def applied(self) -> "Thresholds":
        m = 1.0 + self.slack * 0.19
        return Thresholds(self.ear_drop * m, self.shoulder_width * m,
                          self.shoulder_tilt * m, 0)


@dataclass
class Settings:
    interval_s: int = 600
    cooldown_s: int = 1800
    quiet_start: int = 18
    quiet_end: int = 7
    autostart: bool = False
    camera: int = 0


# ---- detection helpers ---------------------------------------------------

def compute_metrics(lm) -> Metrics | None:
    required = [L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, NOSE, L_EYE, R_EYE]
    confidence = min(lm[i].visibility for i in required)
    if confidence < 0.6:
        return None
    ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
    le, re = lm[L_EAR], lm[R_EAR]
    sw = float(np.hypot(ls.x - rs.x, ls.y - rs.y))
    if sw < 1e-6:
        return None
    return Metrics(
        ear_drop=((le.y + re.y) / 2 - (ls.y + rs.y) / 2) / sw,
        shoulder_width=sw,
        shoulder_tilt=abs(ls.y - rs.y) / sw,
        confidence=float(confidence),
    )


def deltas(m: Metrics, base: Metrics) -> dict:
    return {
        "ear_drop": m.ear_drop - base.ear_drop,
        "shoulder_width": (m.shoulder_width - base.shoulder_width)
                          / max(base.shoulder_width, 1e-6),
        "shoulder_tilt": m.shoulder_tilt - base.shoulder_tilt,
    }


def is_bad(m: Metrics, base: Metrics, th: Thresholds) -> tuple[bool, list[str]]:
    eff = th.applied()
    d = deltas(m, base)
    reasons: list[str] = []
    if d["ear_drop"] > eff.ear_drop:
        reasons.append(f"head forward (+{d['ear_drop']:.2f})")
    if d["shoulder_width"] > eff.shoulder_width:
        reasons.append(f"leaning in (+{d['shoulder_width']*100:.0f}%)")
    if d["shoulder_tilt"] > eff.shoulder_tilt:
        reasons.append(f"shoulders uneven (+{d['shoulder_tilt']:.2f})")
    return (len(reasons) > 0, reasons)


def in_quiet(now: datetime, qs: int, qe: int) -> bool:
    h = now.hour
    if qs == qe:
        return False
    if qs < qe:
        return qs <= h < qe
    return h >= qs or h < qe


# ---- IO ------------------------------------------------------------------

def load_baseline() -> Metrics | None:
    if not BASELINE_PATH.exists():
        return None
    return Metrics(**json.loads(BASELINE_PATH.read_text()))


def save_baseline(b: Metrics):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(asdict(b), indent=2))


def load_thresholds() -> Thresholds:
    if not THRESHOLDS_PATH.exists():
        return Thresholds()
    d = json.loads(THRESHOLDS_PATH.read_text())
    return Thresholds(
        ear_drop=d.get("ear_drop", 0.20),
        shoulder_width=d.get("shoulder_width", 0.15),
        shoulder_tilt=d.get("shoulder_tilt", 0.08),
        slack=int(d.get("slack", 0)),
    )


def save_thresholds(th: Thresholds):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(asdict(th), indent=2))


def load_settings() -> Settings:
    if not SETTINGS_PATH.exists():
        return Settings()
    d = json.loads(SETTINGS_PATH.read_text())
    return Settings(**{**asdict(Settings()), **d})


def save_settings(s: Settings):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(asdict(s), indent=2))


def log_check(status: str, m: Metrics | None = None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    e: dict = {"ts": time.time(), "status": status}
    if m is not None:
        e["ear_drop"] = round(m.ear_drop, 3)
        e["shoulder_width"] = round(m.shoulder_width, 3)
        e["shoulder_tilt"] = round(m.shoulder_tilt, 3)
    with CHECK_LOG.open("a") as f:
        f.write(json.dumps(e) + "\n")


def log_nudge(reasons: list[str]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with NUDGE_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "reasons": reasons}) + "\n")


# ---- autostart (Windows) -------------------------------------------------

def windows_startup_shortcut_path() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft/Windows/Start Menu/Programs/Startup/PostureNudge.lnk"


def set_windows_autostart(enabled: bool, exe_path: Path | None = None) -> str:
    """Create or remove a startup shortcut. Returns a status string."""
    if not IS_WINDOWS:
        return "autostart only supported on Windows"
    p = windows_startup_shortcut_path()
    if p is None:
        return "APPDATA not set"
    if not enabled:
        if p.exists():
            p.unlink()
        return "autostart disabled"
    target = exe_path or Path(sys.executable)
    try:
        # Use PowerShell to create the .lnk; avoids pythoncom dependency.
        import subprocess
        ps = (
            f'$s = (New-Object -ComObject WScript.Shell).CreateShortcut("{p}");'
            f'$s.TargetPath = "{target}";'
            f'$s.WorkingDirectory = "{target.parent}";'
            '$s.Save()'
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            check=True, capture_output=True, timeout=10,
        )
        return f"autostart enabled ({p.name})"
    except Exception as e:
        return f"autostart failed: {e}"


# ---- popup (in-process Tkinter Toplevel) ---------------------------------

def _primary_screen_geom(root: tk.Tk) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) of the primary monitor.

    On Windows the Tk winfo_screen* values match the primary monitor. On Linux
    with multiple displays they can return the bounding box of all monitors,
    which puts overlay windows at the wrong position. Fall back to xrandr in
    that case.
    """
    if sys.platform.startswith("linux"):
        try:
            import subprocess
            out = subprocess.check_output(
                ["xrandr", "--query"], text=True, timeout=1.0,
                stderr=subprocess.DEVNULL,
            )
            import re
            for line in out.split("\n"):
                if " connected primary " in line:
                    m = re.search(r"(\d+)x(\d+)\+(\d+)\+(\d+)", line)
                    if m:
                        w, h, x, y = (int(m.group(i)) for i in (1, 2, 3, 4))
                        return x, y, w, h
        except Exception:
            pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


class CharacterPopup:
    """Slide-up chair popup shown by the main app's Tk root."""

    WIDTH, HEIGHT = 320, 130
    HOLD_MS = 4500
    STEP_PX, TICK_MS = 14, 14
    CHAIR_TARGET_PX = 96  # height to display the chair at

    def __init__(self, root: tk.Tk, title: str, body: str):
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.wm_attributes("-topmost", True)
        try:
            self.win.wm_attributes("-toolwindow", True)  # Windows: no taskbar
        except tk.TclError:
            pass
        self.win.configure(bg=BG)
        # Force size — overrideredirect would otherwise let Tk resize to content.
        self.win.resizable(False, False)
        self.win.minsize(self.WIDTH, self.HEIGHT)
        self.win.maxsize(self.WIDTH, self.HEIGHT)

        sx, sy, sw, sh = _primary_screen_geom(root)
        self.target_x = sx + sw - self.WIDTH
        self.target_y = sy + sh - self.HEIGHT
        self.start_y = sy + sh + 5
        self.cur_y = self.start_y

        accent = tk.Frame(self.win, bg=ACCENT, width=4)
        accent.pack(side="left", fill="y")
        body_frame = tk.Frame(self.win, bg=BG, padx=14, pady=10)
        body_frame.pack(side="left", fill="both", expand=True)

        chair_added = False
        if CHAIR_PNG.exists():
            try:
                from PIL import Image, ImageTk
                im = Image.open(str(CHAIR_PNG))
                ratio = self.CHAIR_TARGET_PX / im.height
                im = im.resize(
                    (int(im.width * ratio), self.CHAIR_TARGET_PX),
                    Image.LANCZOS,
                )
                self.chair = ImageTk.PhotoImage(im)
                tk.Label(body_frame, image=self.chair, bg=BG, borderwidth=0
                         ).pack(side="left", padx=(0, 12))
                chair_added = True
            except Exception:
                pass
        if not chair_added:
            tk.Label(body_frame, text="🪑", bg=BG, fg=FG,
                     font=("Segoe UI", 36)).pack(side="left", padx=(0, 12))

        text_frame = tk.Frame(body_frame, bg=BG)
        text_frame.pack(side="left", anchor="center")
        tk.Label(text_frame, text=title, bg=BG, fg=FG,
                 font=("Segoe UI", 14, "bold"), anchor="w", justify="left"
                 ).pack(anchor="w")
        tk.Label(text_frame, text=body, bg=BG, fg=BLUE,
                 font=("Segoe UI", 10), anchor="w", justify="left",
                 wraplength=self.WIDTH - 130
                 ).pack(anchor="w")

        # Set geometry AFTER packing so widgets don't shrink-wrap the window
        # below our requested size.
        self.win.update_idletasks()
        self.win.geometry(
            f"{self.WIDTH}x{self.HEIGHT}+{self.target_x}+{self.start_y}")
        self.win.after(self.TICK_MS, self._slide_in)

    def _slide_in(self):
        if self.cur_y > self.target_y:
            self.cur_y = max(self.target_y, self.cur_y - self.STEP_PX)
            self.win.geometry(f"{self.WIDTH}x{self.HEIGHT}+{self.target_x}+{self.cur_y}")
            self.win.after(self.TICK_MS, self._slide_in)
        else:
            self.win.after(self.HOLD_MS, self._slide_out)

    def _slide_out(self):
        if self.cur_y < self.start_y:
            self.cur_y = min(self.start_y, self.cur_y + self.STEP_PX)
            self.win.geometry(f"{self.WIDTH}x{self.HEIGHT}+{self.target_x}+{self.cur_y}")
            self.win.after(self.TICK_MS, self._slide_out)
        else:
            try:
                self.win.destroy()
            except tk.TclError:
                pass


# ---- monitor thread ------------------------------------------------------

class Monitor(threading.Thread):
    """Background loop that opens the camera every interval, runs detection,
    and pushes events to a queue read by the GUI thread."""

    daemon = True

    def __init__(self, settings: Settings, evt_q: queue.Queue):
        super().__init__()
        self.settings = settings
        self.evt_q = evt_q
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def _new_pose(self):
        return mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True)

    def _capture_burst(self, pose, cap, want=15) -> Metrics | None:
        samples: list[Metrics] = []
        max_reads = want * 4
        reads = 0
        while len(samples) < want and reads < max_reads and not self.stop_event.is_set():
            reads += 1
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            r = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if r.pose_landmarks:
                m = compute_metrics(r.pose_landmarks.landmark)
                if m is not None:
                    samples.append(m)
        if not samples:
            return None
        return Metrics(
            ear_drop=float(np.median([s.ear_drop for s in samples])),
            shoulder_width=float(np.median([s.shoulder_width for s in samples])),
            shoulder_tilt=float(np.median([s.shoulder_tilt for s in samples])),
            confidence=float(np.mean([s.confidence for s in samples])),
        )

    def run(self):
        last_notified = 0.0
        pose = self._new_pose()
        while not self.stop_event.is_set():
            now = datetime.now()
            if in_quiet(now, self.settings.quiet_start, self.settings.quiet_end):
                log_check("quiet")
                self.evt_q.put(("status", "quiet hours"))
                self._sleep(self.settings.interval_s)
                continue

            baseline = load_baseline()
            th = load_thresholds()
            if baseline is None:
                self.evt_q.put(("status", "no baseline — calibrate"))
                self._sleep(self.settings.interval_s)
                continue

            cap = cv2.VideoCapture(self.settings.camera)
            if not cap.isOpened():
                cap.release()
                log_check("camera_busy")
                self.evt_q.put(("status", "camera busy"))
                self._sleep(self.settings.interval_s)
                continue

            m = self._capture_burst(pose, cap)
            cap.release()

            if m is None:
                log_check("absent")
                self.evt_q.put(("status", "no person detected"))
            else:
                bad, reasons = is_bad(m, baseline, th)
                log_check("bad" if bad else "good", m)
                if bad:
                    self.evt_q.put(("status", "BAD: " + "; ".join(reasons)))
                    if time.time() - last_notified >= self.settings.cooldown_s:
                        log_nudge(reasons)
                        self.evt_q.put(("nudge", reasons))
                        last_notified = time.time()
                else:
                    self.evt_q.put(("status", "OK"))

            self._sleep(self.settings.interval_s)
        try:
            pose.close()
        except Exception:
            pass

    def _sleep(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end and not self.stop_event.is_set():
            time.sleep(0.5)


# ---- calibration / test dialog ------------------------------------------

class CalibrationDialog(tk.Toplevel):
    """Two-phase: capture baseline, then live test with slack slider."""

    CAPTURE_S = 5.0
    TEST_S = 20.0

    def __init__(self, root: tk.Tk, settings: Settings):
        super().__init__(root)
        self.title("Calibrate posture")
        self.configure(bg=BG)
        self.geometry("720x520")
        self.settings = settings
        self.cap: cv2.VideoCapture | None = None
        self.pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True)
        self.samples: list[Metrics] = []
        self.phase = "capture"  # "capture" | "test" | "done"
        self.phase_end = time.time() + self.CAPTURE_S
        self.baseline: Metrics | None = None
        self.live_th: Thresholds | None = None
        self.saved_th = load_thresholds()
        self.slack_var = tk.IntVar(value=self.saved_th.slack)

        # Top: phase banner (big, color-coded)
        self.banner_var = tk.StringVar(value="CALIBRATING — hold your good posture")
        self.banner = tk.Label(self, textvariable=self.banner_var, bg=BG,
                               fg=YELLOW, font=("Segoe UI", 16, "bold"))
        self.banner.pack(pady=(8, 4))

        self.video_label = tk.Label(self, bg=BG)
        self.video_label.pack(fill="both", expand=True, padx=8, pady=4)

        # Slack slider row (only used in test phase but visible from the start)
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=8, pady=4)
        tk.Label(bar, text="slack:", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).pack(side="left")
        self.slack_label_var = tk.StringVar(value=f"{self.saved_th.slack}/100")
        tk.Label(bar, textvariable=self.slack_label_var, bg=BG, fg=FG,
                 width=8).pack(side="left", padx=(4, 8))
        ttk.Scale(bar, from_=0, to=100, orient="horizontal",
                  variable=self.slack_var,
                  command=lambda v: self.slack_label_var.set(
                      f"{int(float(v))}/100")
                  ).pack(side="left", fill="x", expand=True, padx=8)

        # Bottom: status line (countdown / samples / details)
        self.status_var = tk.StringVar(value="Camera starting…")
        tk.Label(self, textvariable=self.status_var, bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(pady=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(50, self.tick)

    def _open_camera(self) -> bool:
        # Re-create on failure: a VideoCapture that failed to open stays
        # closed forever otherwise, so we'd retry-loop with no chance of
        # success. Re-create each retry until something else releases the
        # camera (e.g. a competing periodic-check process finishes).
        if self.cap is None or not self.cap.isOpened():
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.settings.camera)
        return self.cap.isOpened()

    def tick(self):
        if not self._open_camera():
            self.banner_var.set("WAITING FOR CAMERA")
            self.banner.configure(fg=YELLOW)
            self.status_var.set("Camera busy — close other webcam apps. Retrying…")
            self.after(500, self.tick)
            return

        ok, frame = self.cap.read()  # type: ignore[union-attr]
        if not ok:
            self.after(50, self.tick)
            return

        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        m = (compute_metrics(result.pose_landmarks.landmark)
             if result.pose_landmarks else None)
        remaining = max(0, self.phase_end - time.time())

        if self.phase == "capture":
            if m is not None:
                self.samples.append(m)
            face_seen = m is not None
            self.banner_var.set(
                f"CALIBRATING — hold your good posture  ({remaining:.1f}s)")
            self.banner.configure(fg=YELLOW if face_seen else RED)
            self.status_var.set(
                f"samples captured: {len(self.samples)}    "
                f"face: {'detected ✓' if face_seen else 'NOT detected — sit in view'}")
            if remaining <= 0:
                self._finish_capture()
        elif self.phase == "test":
            slack = self.slack_var.get()
            self.live_th = Thresholds(
                ear_drop=self.saved_th.ear_drop,
                shoulder_width=self.saved_th.shoulder_width,
                shoulder_tilt=self.saved_th.shoulder_tilt,
                slack=slack,
            )
            mult = 1 + slack * 0.19
            if m is not None and self.baseline is not None:
                bad, reasons = is_bad(m, self.baseline, self.live_th)
                if bad:
                    self.banner_var.set(f"BAD POSTURE — {'; '.join(reasons[:2])}")
                    self.banner.configure(fg=RED)
                else:
                    self.banner_var.set("GOOD POSTURE ✓")
                    self.banner.configure(fg=GREEN)
            else:
                self.banner_var.set("FACE NOT DETECTED — sit in view")
                self.banner.configure(fg=YELLOW)
            self.status_var.set(
                f"test — try slouching / leaning  ({remaining:.1f}s)    "
                f"slack {slack}/100 = {mult:.1f}× more lenient    "
                f"(slider auto-saves on close)")
            if remaining <= 0:
                self._finish_test()
                return

        self._render_frame(frame, result)
        self.after(50, self.tick)

    def _render_frame(self, frame, result):
        h, w = frame.shape[:2]
        if result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        # Resize for display (max 640 wide)
        max_w = 640
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk  # lazy: PIL only needed in the dialog
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # type: ignore[attr-defined]

    def _finish_capture(self):
        if len(self.samples) < 5:
            self.banner_var.set("NOT ENOUGH SAMPLES — sit in view")
            self.banner.configure(fg=RED)
            self.status_var.set(
                f"only {len(self.samples)} valid samples — retrying capture")
            self.samples = []
            self.phase_end = time.time() + self.CAPTURE_S
            return
        self.baseline = Metrics(
            ear_drop=float(np.median([s.ear_drop for s in self.samples])),
            shoulder_width=float(np.median([s.shoulder_width for s in self.samples])),
            shoulder_tilt=float(np.median([s.shoulder_tilt for s in self.samples])),
            confidence=float(np.mean([s.confidence for s in self.samples])),
        )
        save_baseline(self.baseline)
        # Brief celebratory banner before transitioning to test phase
        self.banner_var.set(
            f"BASELINE SAVED ({len(self.samples)} samples, "
            f"conf {self.baseline.confidence:.0%})")
        self.banner.configure(fg=GREEN)
        self.status_var.set("now slouch / lean to verify it responds…")
        self.phase = "test"
        self.phase_end = time.time() + self.TEST_S

    def _finish_test(self):
        if self.live_th is not None:
            save_thresholds(self.live_th)
        self.close()

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
            self.pose.close()
        except Exception:
            pass
        self.destroy()


# ---- stats tab -----------------------------------------------------------

class StatsView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.fig = Figure(figsize=(6, 4.5), facecolor=BG, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.refresh_btn = ttk.Button(self, text="Refresh", command=self.refresh)
        self.refresh_btn.pack(pady=4)
        self.refresh()

    def refresh(self):
        self.fig.clear()
        today = datetime.now().date()
        days_back = 7
        days = [today - timedelta(days=i) for i in range(days_back - 1, -1, -1)]

        counts = [[{"good": 0, "bad": 0} for _ in range(24)] for _ in range(days_back)]
        if CHECK_LOG.exists():
            for line in CHECK_LOG.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = e.get("status", "")
                if s not in ("good", "bad"):
                    continue
                ts = datetime.fromtimestamp(e["ts"])
                if ts.date() not in days:
                    continue
                counts[days.index(ts.date())][ts.hour][s] += 1

        WORK_FROM, WORK_TO = 7, 18
        matrix = np.full((days_back, WORK_TO - WORK_FROM + 1), np.nan)
        for di in range(days_back):
            for hi in range(WORK_FROM, WORK_TO + 1):
                c = counts[di][hi]
                if c["good"] + c["bad"] >= 1:
                    matrix[di, hi - WORK_FROM] = c["bad"] / (c["good"] + c["bad"])

        day_totals = []
        day_fracs = []
        for di in range(days_back):
            good = sum(counts[di][h]["good"] for h in range(24))
            bad = sum(counts[di][h]["bad"] for h in range(24))
            day_totals.append(good + bad)
            day_fracs.append(bad / (good + bad) if (good + bad) > 0 else float("nan"))

        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.7)
        ax1 = self.fig.add_subplot(gs[0])
        ax2 = self.fig.add_subplot(gs[1])
        cmap = LinearSegmentedColormap.from_list("posture", [GREEN, YELLOW, RED])
        cmap.set_bad(MUTED)
        ax1.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
                   vmin=0, vmax=1, interpolation="nearest")
        ax1.set_title("Bad-posture % by hour (last 7d)", color=FG, fontsize=10,
                      loc="left")
        hours = list(range(WORK_FROM, WORK_TO + 1))
        tick_idx = list(range(0, len(hours), 2))
        ax1.set_xticks(tick_idx)
        ax1.set_xticklabels([f"{hours[i]:02d}" for i in tick_idx], color=FG)
        ax1.set_yticks(range(days_back))
        ax1.set_yticklabels([d.strftime("%a") for d in days], color=FG)
        ax1.set_facecolor(MUTED)
        for s in ax1.spines.values():
            s.set_color(FG)
        ax1.tick_params(colors=FG, length=0)

        bars = []
        heights = []
        MIN = 6
        for i, f in enumerate(day_fracs):
            if np.isnan(f) or day_totals[i] < MIN:
                bars.append(MUTED)
                heights.append(0)
            else:
                bars.append(cmap(f))
                heights.append(f * 100)
        ax2.bar(range(days_back), heights, color=bars, width=0.85)
        ax2.set_facecolor(BG)
        ax2.set_title("Bad % per day", color=FG, fontsize=10, loc="left")
        ax2.set_xticks(range(days_back))
        ax2.set_xticklabels([d.strftime("%a") for d in days], color=FG)
        ax2.set_ylim(0, 100)
        ax2.set_yticks([0, 50, 100])
        ax2.set_yticklabels(["0", "50", "100"], color=FG)
        ax2.tick_params(colors=FG, length=0)
        for s in ax2.spines.values():
            s.set_color(FG)
        ax2.grid(True, axis="y", alpha=0.18, color=FG)
        ax2.set_axisbelow(True)
        self.canvas.draw_idle()


# ---- main app ------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Posture Nudge")
        self.geometry("760x600")
        self.configure(bg=BG)
        self.evt_q: queue.Queue = queue.Queue()
        self.monitor: Monitor | None = None
        self.settings = load_settings()

        try:
            style = ttk.Style(self)
            style.theme_use("clam")
            style.configure(".", background=BG, foreground=FG, fieldbackground=BG)
            style.configure("TNotebook", background=BG, borderwidth=0)
            style.configure("TNotebook.Tab", background=MUTED, foreground=FG,
                            padding=(12, 6))
            style.configure("TButton", background=MUTED, foreground=FG, padding=6)
            style.configure("TLabel", background=BG, foreground=FG)
            style.configure("TFrame", background=BG)
        except tk.TclError:
            pass

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self._build_monitor_tab()
        self._build_stats_tab()
        self._build_settings_tab()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(200, self._drain_events)

        # Auto-start monitor if baseline exists
        if BASELINE_PATH.exists():
            self.start_monitor()

    def _build_monitor_tab(self):
        f = tk.Frame(self.nb, bg=BG, padx=12, pady=12)
        self.nb.add(f, text="Monitor")
        self.status_var = tk.StringVar(value="Stopped")
        tk.Label(f, text="Status:", bg=BG, fg=FG,
                 font=("Segoe UI", 11)).pack(anchor="w")
        tk.Label(f, textvariable=self.status_var, bg=BG, fg=BLUE,
                 font=("Segoe UI", 14)).pack(anchor="w", pady=(0, 12))
        ttk.Button(f, text="Calibrate / Test", command=self.open_calibrate
                   ).pack(fill="x", pady=4)
        self.toggle_btn = ttk.Button(f, text="Start monitoring",
                                     command=self.toggle_monitor)
        self.toggle_btn.pack(fill="x", pady=4)
        ttk.Button(f, text="Test popup now", command=self.fire_test_popup
                   ).pack(fill="x", pady=4)
        msg = ("Monitor checks your posture every "
               f"{self.settings.interval_s // 60} minutes. "
               f"Quiet between {self.settings.quiet_start:02d}:00 and "
               f"{self.settings.quiet_end:02d}:00.")
        tk.Label(f, text=msg, bg=BG, fg=MUTED, wraplength=700,
                 justify="left").pack(anchor="w", pady=(20, 0))
        if BASELINE_PATH.exists():
            b = load_baseline()
            tk.Label(f, text=f"Baseline saved (confidence "
                             f"{(b.confidence if b else 0):.0%})",
                     bg=BG, fg=GREEN).pack(anchor="w", pady=4)
        else:
            tk.Label(f, text="No baseline — click Calibrate to get started.",
                     bg=BG, fg=YELLOW).pack(anchor="w", pady=4)

    def _build_stats_tab(self):
        self.stats_view = StatsView(self.nb)
        self.nb.add(self.stats_view, text="Stats")

    def _build_settings_tab(self):
        f = tk.Frame(self.nb, bg=BG, padx=12, pady=12)
        self.nb.add(f, text="Settings")

        def add_row(label: str, var: tk.Variable, kind="entry"):
            row = tk.Frame(f, bg=BG)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, bg=BG, fg=FG, width=20,
                     anchor="w").pack(side="left")
            if kind == "entry":
                ttk.Entry(row, textvariable=var, width=12).pack(side="left")
            elif kind == "check":
                ttk.Checkbutton(row, variable=var).pack(side="left")
            return row

        self.s_interval = tk.IntVar(value=self.settings.interval_s // 60)
        self.s_cooldown = tk.IntVar(value=self.settings.cooldown_s // 60)
        self.s_qstart = tk.IntVar(value=self.settings.quiet_start)
        self.s_qend = tk.IntVar(value=self.settings.quiet_end)
        self.s_autostart = tk.BooleanVar(value=self.settings.autostart)
        self.s_camera = tk.IntVar(value=self.settings.camera)

        add_row("Check interval (min)", self.s_interval)
        add_row("Cooldown between nudges (min)", self.s_cooldown)
        add_row("Quiet hours start (0–23)", self.s_qstart)
        add_row("Quiet hours end (0–23)", self.s_qend)
        add_row("Camera index (0/1/…)", self.s_camera)
        add_row("Start with Windows", self.s_autostart, kind="check")

        # Slack slider (more prominent than the one inside Calibrate dialog).
        cur_th = load_thresholds()
        self.s_slack = tk.IntVar(value=cur_th.slack)
        slack_row = tk.Frame(f, bg=BG)
        slack_row.pack(fill="x", pady=(16, 4))
        tk.Label(slack_row, text="Slack (0=strict, 100=very lenient)",
                 bg=BG, fg=FG, anchor="w", width=32
                 ).pack(side="left")
        self.s_slack_label = tk.StringVar(value=f"{cur_th.slack}/100")
        tk.Label(slack_row, textvariable=self.s_slack_label, bg=BG, fg=FG,
                 width=10).pack(side="left")
        ttk.Scale(slack_row, from_=0, to=100, orient="horizontal",
                  variable=self.s_slack,
                  command=lambda v: self.s_slack_label.set(
                      f"{int(float(v))}/100")
                  ).pack(side="left", fill="x", expand=True, padx=8)

        self.autostart_status = tk.StringVar(value="")
        tk.Label(f, textvariable=self.autostart_status, bg=BG, fg=MUTED
                 ).pack(anchor="w", pady=(8, 0))

        ttk.Button(f, text="Save settings", command=self.save_settings_clicked
                   ).pack(pady=12, anchor="w")

    # --- actions ---

    def open_calibrate(self):
        if self.monitor is not None:
            self.stop_monitor()
        CalibrationDialog(self, self.settings)
        # Wait for it to close (non-blocking; user closes dialog manually)
        # Refresh status after dialog (we just trust user to start monitor)

    def toggle_monitor(self):
        if self.monitor is None:
            self.start_monitor()
        else:
            self.stop_monitor()

    def start_monitor(self):
        if not BASELINE_PATH.exists():
            messagebox.showinfo("Posture Nudge",
                                "Calibrate first (Monitor tab → Calibrate).")
            return
        self.monitor = Monitor(self.settings, self.evt_q)
        self.monitor.start()
        self.toggle_btn.configure(text="Stop monitoring")
        self.status_var.set("monitoring…")

    def stop_monitor(self):
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor = None
        self.toggle_btn.configure(text="Start monitoring")
        self.status_var.set("Stopped")

    def fire_test_popup(self):
        CharacterPopup(self, "Sit up!", "test popup")

    def save_settings_clicked(self):
        self.settings.interval_s = max(60, int(self.s_interval.get()) * 60)
        self.settings.cooldown_s = max(60, int(self.s_cooldown.get()) * 60)
        self.settings.quiet_start = int(self.s_qstart.get()) % 24
        self.settings.quiet_end = int(self.s_qend.get()) % 24
        self.settings.camera = max(0, int(self.s_camera.get()))
        self.settings.autostart = bool(self.s_autostart.get())
        save_settings(self.settings)
        # Persist the slack slider too.
        th = load_thresholds()
        th.slack = max(0, min(100, int(self.s_slack.get())))
        save_thresholds(th)
        # Apply autostart
        if IS_WINDOWS:
            exe = Path(sys.executable) if IS_FROZEN else None
            self.autostart_status.set(set_windows_autostart(self.settings.autostart, exe))
        else:
            self.autostart_status.set("autostart only on Windows")
        # Restart monitor with new settings if running
        if self.monitor is not None:
            self.stop_monitor()
            self.start_monitor()
        messagebox.showinfo("Posture Nudge", "Settings saved.")

    # --- event drain ---

    def _drain_events(self):
        try:
            while True:
                kind, payload = self.evt_q.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "nudge":
                    CharacterPopup(self, "Sit up!", "; ".join(payload[:2]))
        except queue.Empty:
            pass
        self.after(500, self._drain_events)

    def on_close(self):
        if self.monitor is not None:
            self.stop_monitor()
        self.destroy()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
