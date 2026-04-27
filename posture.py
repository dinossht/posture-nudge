#!/usr/bin/env python3
"""Posture nudge — webcam-based posture monitoring using MediaPipe Pose.

Subcommands:
  calibrate        — capture your good-posture baseline (do this once)
  snap-baseline    — quick re-baseline (for chair/screen changes)
  record           — capture labeled samples for each of a set of postures
  analyze          — summarize samples and write per-metric thresholds
  visualize        — live overlay of landmarks + metrics (no alerts)
  monitor          — periodic posture check, send desktop alerts when you slouch
  stats            — summarize recent nudges from the nudge log
  digest           — one-shot stats summary as a desktop notification
  install-service  — systemd user unit so monitor runs on login
  install-digest   — systemd user timer that runs `digest` weekly
  install-shortcut — GNOME keyboard shortcut that invokes snap-baseline
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

CONFIG_DIR = Path.home() / ".config" / "posture_nudge"
BASELINE_PATH = CONFIG_DIR / "baseline.json"
SAMPLES_PATH = CONFIG_DIR / "samples.jsonl"
THRESHOLDS_PATH = CONFIG_DIR / "thresholds.json"
NUDGE_LOG = CONFIG_DIR / "nudge_log.jsonl"
CHECK_LOG = CONFIG_DIR / "check_log.jsonl"  # one line per monitor check
PLOT_PATH = CONFIG_DIR / "plot.png"

POSTURE_SEQUENCE: list[tuple[str, str]] = [
    ("good_normal",  "Sit upright at your normal laptop distance."),
    ("good_back",    "Sit upright but lean back (further from screen)."),
    ("slouch_mild",  "Slouch the way you normally drift into."),
    ("slouch_heavy", "Slouch heavily — shoulders rolled forward, head dropped."),
    ("head_forward", "Spine upright but crane your head forward toward screen."),
    ("lean_in",      "Lean whole upper body in close to the screen."),
    ("tilt_left",    "Drop your left shoulder / lean on left elbow."),
    ("tilt_right",   "Drop your right shoulder / lean on right elbow."),
]

# MediaPipe pose landmark indices
NOSE = 0
L_EYE_INNER, L_EYE, L_EYE_OUTER = 1, 2, 3
R_EYE_INNER, R_EYE, R_EYE_OUTER = 4, 5, 6
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12

METRIC_NAMES = ("ear_drop", "shoulder_width", "shoulder_tilt")


@dataclass
class Metrics:
    ear_drop: float        # (ear_mid_y - shoulder_mid_y) / shoulder_width — less negative ⇒ head forward/down
    shoulder_width: float  # shoulder width in normalized image coords — larger ⇒ leaning toward screen
    shoulder_tilt: float   # |L_y - R_y| / shoulder_width — larger ⇒ asymmetric/leaning to a side
    confidence: float


@dataclass
class Thresholds:
    """Per-metric threshold deltas above baseline. Used in `is_bad_posture`."""
    ear_drop: float = 0.20
    shoulder_width: float = 0.15  # relative (15% wider than baseline)
    shoulder_tilt: float = 0.08


def compute_metrics(lm) -> Metrics | None:
    # Require shoulders + ears AND face landmarks (nose + at least one eye) so
    # we don't measure when the user is looking away, head out of frame, etc.
    required = [L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, NOSE, L_EYE, R_EYE]
    confidence = min(lm[i].visibility for i in required)
    if confidence < 0.6:
        return None

    ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
    le, re = lm[L_EAR], lm[R_EAR]
    shoulder_width = float(np.hypot(ls.x - rs.x, ls.y - rs.y))
    if shoulder_width < 1e-6:
        return None

    shoulder_mid_y = (ls.y + rs.y) / 2
    ear_mid_y = (le.y + re.y) / 2
    return Metrics(
        ear_drop=(ear_mid_y - shoulder_mid_y) / shoulder_width,
        shoulder_width=shoulder_width,
        shoulder_tilt=abs(ls.y - rs.y) / shoulder_width,
        confidence=float(confidence),
    )


def compute_deltas(m: Metrics, base: Metrics) -> dict[str, float]:
    """Return per-metric deviation from baseline. Positive = 'worse' direction."""
    return {
        "ear_drop":       m.ear_drop - base.ear_drop,
        "shoulder_width": (m.shoulder_width - base.shoulder_width) / max(base.shoulder_width, 1e-6),
        "shoulder_tilt":  m.shoulder_tilt - base.shoulder_tilt,
    }


def is_bad_posture(m: Metrics, base: Metrics, th: Thresholds) -> tuple[bool, list[str]]:
    d = compute_deltas(m, base)
    reasons = []
    if d["ear_drop"] > th.ear_drop:
        reasons.append(f"head forward (+{d['ear_drop']:.2f})")
    if d["shoulder_width"] > th.shoulder_width:
        reasons.append(f"leaning in (+{d['shoulder_width']*100:.0f}%)")
    if d["shoulder_tilt"] > th.shoulder_tilt:
        reasons.append(f"shoulders uneven (+{d['shoulder_tilt']:.2f})")
    return (len(reasons) > 0, reasons)


LID_STATE_PATH = Path("/proc/acpi/button/lid/LID0/state")


def lid_state() -> str:
    """Return 'open', 'closed', or 'unknown' (sysfs file missing on some hw)."""
    try:
        text = LID_STATE_PATH.read_text()
        return "closed" if "closed" in text else ("open" if "open" in text else "unknown")
    except FileNotFoundError:
        return "unknown"


def open_camera(index: int) -> cv2.VideoCapture | None:
    """Open camera; returns None if unavailable (so callers can retry)."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def camera_blocked(cap: cv2.VideoCapture, frames: int = 3) -> bool:
    """Read a few frames and return True if they're all near-black (covered/shuttered)."""
    means = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            continue
        means.append(float(np.mean(frame)))
    if not means:
        return True  # couldn't read at all
    return max(means) < 5.0  # all frames near-black


def load_baseline() -> Metrics:
    if not BASELINE_PATH.exists():
        sys.exit(f"No baseline at {BASELINE_PATH}. Run `calibrate` first.")
    return Metrics(**json.loads(BASELINE_PATH.read_text()))


def load_thresholds() -> Thresholds:
    if THRESHOLDS_PATH.exists():
        data = json.loads(THRESHOLDS_PATH.read_text())
        return Thresholds(**{k: data[k] for k in METRIC_NAMES if k in data})
    return Thresholds()


def save_thresholds(t: Thresholds):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(asdict(t), indent=2))


def draw_overlay(frame, lm, metrics: Metrics | None, baseline: Metrics | None, th: Thresholds):
    if lm is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, lm, mp.solutions.pose.POSE_CONNECTIONS
        )
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if metrics is None:
        lines.append(("(no pose / low confidence)", (0, 180, 255)))
    else:
        bad, reasons = (False, [])
        if baseline is not None:
            bad, reasons = is_bad_posture(metrics, baseline, th)
        color = (0, 0, 255) if bad else (0, 255, 0)
        status = "BAD" if bad else "OK"
        lines.append((f"{status}  ear_drop={metrics.ear_drop:+.2f}  "
                      f"sw={metrics.shoulder_width:.2f}  tilt={metrics.shoulder_tilt:.2f}  "
                      f"conf={metrics.confidence:.2f}", color))
        if baseline is not None:
            lines.append((f"base  ear_drop={baseline.ear_drop:+.2f}  "
                          f"sw={baseline.shoulder_width:.2f}  tilt={baseline.shoulder_tilt:.2f}   "
                          f"th={th.ear_drop:.2f}/{th.shoulder_width:.2f}/{th.shoulder_tilt:.2f}",
                          (200, 200, 200)))
            if reasons:
                lines.append((" + " + "; ".join(reasons), (0, 0, 255)))
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _new_pose():
    return mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True)


def cmd_calibrate(args):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Sit up straight. Capturing for {args.duration:.0f}s in 3s…")
    time.sleep(3)

    pose = _new_pose()
    cap = open_camera(args.camera)
    if cap is None:
        sys.exit(f"Could not open camera {args.camera}.")
    samples: list[Metrics] = []
    end = time.time() + args.duration
    th = load_thresholds()
    try:
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                continue
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            m = None
            if result.pose_landmarks:
                m = compute_metrics(result.pose_landmarks.landmark)
                if m is not None:
                    samples.append(m)
            draw_overlay(frame, result.pose_landmarks, m, None, th)
            cv2.putText(frame, f"calibrating… {max(0, end - time.time()):.1f}s",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
            cv2.imshow("posture-nudge calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

    if not samples:
        sys.exit("No valid samples captured. Make sure shoulders + ears are visible.")

    baseline = Metrics(
        ear_drop=float(np.median([s.ear_drop for s in samples])),
        shoulder_width=float(np.median([s.shoulder_width for s in samples])),
        shoulder_tilt=float(np.median([s.shoulder_tilt for s in samples])),
        confidence=float(np.mean([s.confidence for s in samples])),
    )
    BASELINE_PATH.write_text(json.dumps(asdict(baseline), indent=2))
    print(f"Saved baseline from {len(samples)} samples:")
    print(json.dumps(asdict(baseline), indent=2))
    print(f"Written to {BASELINE_PATH}")


def cmd_snap_baseline(args):
    """Quick re-calibration: capture a few seconds of the current posture and save as baseline.

    Triggered by a keyboard shortcut after chair/screen/desk changes. By default
    pops up a small live preview so the user can see they're in frame and the
    pose is being detected. Pass --no-gui for headless mode.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Lid closed = no point trying to grab the webcam.
    if lid_state() == "closed":
        notify("posture-nudge", "Laptop lid is closed — open it and try again")
        sys.exit(1)

    # Monitor service holds the camera for ~0.5s at a time; retry for a few seconds.
    start = time.time()
    cap: cv2.VideoCapture | None = None
    while cap is None and time.time() - start < 5:
        cap = open_camera(args.camera)
        if cap is None:
            time.sleep(0.3)
    if cap is None:
        notify("posture-nudge", "Camera busy — try again in a moment")
        sys.exit(1)

    # Camera opened but might be physically blocked (privacy shutter / camera-off
    # function key / lens cover). Detect by checking if frames are all-black.
    if camera_blocked(cap):
        cap.release()
        if lid_state() == "closed":
            notify("posture-nudge", "Laptop lid closed — open it and try again")
        else:
            notify("posture-nudge",
                   "Camera is closed (privacy shutter or Fn-key) — enable it and retry")
        sys.exit(1)

    pose = _new_pose()
    th = load_thresholds() if THRESHOLDS_PATH.exists() else Thresholds()

    # Warmup: read a few frames so MediaPipe's person detector has time to lock on
    # before we start counting capture seconds.
    for _ in range(8):
        ok, frame = cap.read()
        if ok:
            pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    samples: list[Metrics] = []
    end = time.time() + args.duration
    aborted = False
    win_title = "posture-nudge"

    try:
        # Phase 1: calibration capture
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                continue
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            m = None
            if result.pose_landmarks:
                m = compute_metrics(result.pose_landmarks.landmark)
                if m is not None:
                    samples.append(m)

            if not args.no_gui:
                draw_overlay(frame, result.pose_landmarks, m, None, th)
                remaining = max(0, end - time.time())
                cv2.putText(frame, f"Hold your good posture  {remaining:.1f}s",
                            (20, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f"samples: {len(samples)}",
                            (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.imshow(win_title, frame)
                cv2.setWindowProperty(win_title, cv2.WND_PROP_TOPMOST, 1)
                if cv2.waitKey(1) & 0xFF == 27:
                    aborted = True
                    break

        if aborted:
            return

        if len(samples) < 5:
            return  # handled outside finally for proper notification

        # Save baseline now so the test phase uses the just-captured values.
        baseline = Metrics(
            ear_drop=float(np.median([s.ear_drop for s in samples])),
            shoulder_width=float(np.median([s.shoulder_width for s in samples])),
            shoulder_tilt=float(np.median([s.shoulder_tilt for s in samples])),
            confidence=float(np.mean([s.confidence for s in samples])),
        )
        BASELINE_PATH.write_text(json.dumps(asdict(baseline), indent=2))
        notify("posture-nudge",
               f"Baseline updated ({len(samples)} samples, conf {baseline.confidence:.0%})")
        print(f"Saved baseline to {BASELINE_PATH}")
        print(json.dumps(asdict(baseline), indent=2))

        # Phase 2: live test with one "slack" slider (skip in --no-gui)
        if not args.no_gui and args.test_seconds > 0:
            saved_th = Thresholds(
                ear_drop=th.ear_drop, shoulder_width=th.shoulder_width,
                shoulder_tilt=th.shoulder_tilt,
            )
            # One slider, 0..100. Multiplier scales 1x..20x of the saved values
            # (linear), so at 100 every threshold is well above any real slouch
            # and effectively reads "good posture" all the time.
            cv2.createTrackbar("slack", win_title, 0, 100, lambda v: None)

            test_end = time.time() + args.test_seconds
            saved_flash_until = 0.0
            live_th = saved_th
            while time.time() < test_end:
                ok, frame = cap.read()
                if not ok:
                    continue
                slack = cv2.getTrackbarPos("slack", win_title)
                mult = 1.0 + slack * 0.19  # 0 -> 1x, 100 -> 20x
                live_th = Thresholds(
                    ear_drop=saved_th.ear_drop * mult,
                    shoulder_width=saved_th.shoulder_width * mult,
                    shoulder_tilt=saved_th.shoulder_tilt * mult,
                )
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                m = compute_metrics(result.pose_landmarks.landmark) if result.pose_landmarks else None
                draw_overlay(frame, result.pose_landmarks, m, baseline, live_th)

                remaining = max(0, test_end - time.time())
                cv2.putText(frame, f"Slouch / lean to test — {remaining:.1f}s",
                            (20, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                hint = f"Slack {slack}/100 ({mult:.1f}x)  •  auto-saves on close  •  ESC = quit"
                cv2.putText(frame, hint, (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                if time.time() < saved_flash_until:
                    cv2.putText(frame, "SAVED", (frame.shape[1] - 130, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                cv2.imshow(win_title, frame)
                cv2.setWindowProperty(win_title, cv2.WND_PROP_TOPMOST, 1)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            # Auto-save the final slack value on close.
            save_thresholds(live_th)
            notify("posture-nudge",
                   f"Thresholds saved (slack {slack}/100, {mult:.1f}x)")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

    if aborted:
        notify("posture-nudge", "Snap cancelled")
        sys.exit(1)

    if len(samples) < 5:
        notify("posture-nudge",
               f"Snap failed — only {len(samples)} valid samples. Make sure your "
               f"shoulders + ears are in view and try again.")
        sys.exit(1)


def cmd_install_shortcut(args):
    """Register a GNOME custom keybinding that invokes snap-baseline."""
    schema = "org.gnome.settings-daemon.plugins.media-keys"
    key_slug = "posture-nudge-snap"
    key_path = f"/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/{key_slug}/"
    binding_schema = f"org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:{key_path}"

    python_bin = Path(sys.executable)
    script = Path(__file__).resolve()
    command = f"{python_bin} {script} snap-baseline"

    # Read existing list of custom keybinding paths.
    raw = subprocess.check_output(["gsettings", "get", schema, "custom-keybindings"]).decode().strip()
    if raw in ("@as []", "[]"):
        paths = []
    else:
        # Format is like ['p1', 'p2']. Parse with ast.
        import ast
        paths = ast.literal_eval(raw)

    if key_path not in paths:
        paths.append(key_path)
        # gsettings wants a GVariant-style array-of-string literal.
        as_literal = "[" + ", ".join(f"'{p}'" for p in paths) + "]"
        subprocess.run(["gsettings", "set", schema, "custom-keybindings", as_literal], check=True)

    subprocess.run(["gsettings", "set", binding_schema, "name",
                    "Posture nudge: snap baseline"], check=True)
    subprocess.run(["gsettings", "set", binding_schema, "command", command], check=True)
    subprocess.run(["gsettings", "set", binding_schema, "binding", args.binding], check=True)

    print(f"Bound {args.binding} → {command}")
    print("Press the shortcut whenever your seat/screen changes to re-calibrate.")
    print(f"To undo: gsettings reset-recursively {binding_schema!r}")


def cmd_visualize(args):
    baseline = load_baseline() if BASELINE_PATH.exists() else None
    th = load_thresholds()
    pose = _new_pose()
    cap = open_camera(args.camera)
    if cap is None:
        sys.exit(f"Could not open camera {args.camera}. Is another app using it?")
    print(f"Thresholds: ear={th.ear_drop} width={th.shoulder_width} tilt={th.shoulder_tilt}")
    print("ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            m = compute_metrics(result.pose_landmarks.landmark) if result.pose_landmarks else None
            draw_overlay(frame, result.pose_landmarks, m, baseline, th)
            cv2.imshow("posture-nudge visualize (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()


def capture_samples(seconds: float, camera: int, label: str | None,
                    window_title: str, instruction: str | None) -> list[Metrics]:
    pose = _new_pose()
    cap = open_camera(camera)
    if cap is None:
        sys.exit(f"Could not open camera {camera}.")
    samples: list[Metrics] = []
    end = time.time() + seconds
    th = Thresholds()
    try:
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                continue
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            m = None
            if result.pose_landmarks:
                m = compute_metrics(result.pose_landmarks.landmark)
                if m is not None:
                    samples.append(m)
            draw_overlay(frame, result.pose_landmarks, m, None, th)
            remaining = max(0, end - time.time())
            footer = f"{label or 'capturing'}: {remaining:.1f}s"
            cv2.putText(frame, footer, (20, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if instruction:
                cv2.putText(frame, instruction, (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
    return samples


def cmd_record(args):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and SAMPLES_PATH.exists():
        SAMPLES_PATH.unlink()
    sequence = POSTURE_SEQUENCE if not args.label else [
        (args.label, f"Hold: {args.label}")
    ]
    print("We'll record samples for each posture below.")
    for label, _ in sequence:
        print(f"  • {label}")
    print(f"\nEach: {args.prep}s prep + {args.duration}s capture.")
    print(f"Samples append to {SAMPLES_PATH}")
    print(f"\nStarting in 5s — get ready for the first posture ({sequence[0][0]}).")
    time.sleep(5)

    with SAMPLES_PATH.open("a") as f:
        for label, instruction in sequence:
            print(f"\n— {label} —\n  {instruction}")
            for i in range(int(args.prep), 0, -1):
                print(f"  starting in {i}…", end="\r", flush=True)
                time.sleep(1)
            print(" " * 30, end="\r")
            samples = capture_samples(
                args.duration, args.camera, label,
                window_title=f"record: {label}",
                instruction=instruction,
            )
            for s in samples:
                f.write(json.dumps({"label": label, "ts": time.time(), **asdict(s)}) + "\n")
            print(f"  captured {len(samples)} samples")
    print(f"\nDone. Run `python posture.py analyze` to pick thresholds.")


def cmd_analyze(args):
    if not SAMPLES_PATH.exists():
        sys.exit(f"No samples at {SAMPLES_PATH}. Run `record` first.")

    by_label: dict[str, list[Metrics]] = defaultdict(list)
    for line in SAMPLES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        by_label[d["label"]].append(Metrics(
            ear_drop=d["ear_drop"], shoulder_width=d["shoulder_width"],
            shoulder_tilt=d["shoulder_tilt"], confidence=d["confidence"],
        ))

    if args.refit_baseline:
        good_samples = [s for lab, ss in by_label.items() if lab.startswith("good_") for s in ss]
        if not good_samples:
            sys.exit("No good_* samples to refit from.")
        baseline = Metrics(
            ear_drop=float(np.median([s.ear_drop for s in good_samples])),
            shoulder_width=float(np.median([s.shoulder_width for s in good_samples])),
            shoulder_tilt=float(np.median([s.shoulder_tilt for s in good_samples])),
            confidence=float(np.mean([s.confidence for s in good_samples])),
        )
        BASELINE_PATH.write_text(json.dumps(asdict(baseline), indent=2))
        print(f"Refit baseline from {len(good_samples)} good samples → {BASELINE_PATH}\n")
    else:
        baseline = load_baseline()

    def deltas_for(label: str, metric: str) -> list[float]:
        return [compute_deltas(s, baseline)[metric] for s in by_label.get(label, [])]

    good_prefixes = ("good_",)
    bad_prefixes = ("slouch_", "head_forward", "lean_in", "tilt_")

    print(f"Baseline: ear_drop={baseline.ear_drop:+.2f} sw={baseline.shoulder_width:.2f} "
          f"tilt={baseline.shoulder_tilt:.2f}\n")
    print(f"{'label':<15} {'n':>4}  {'metric':<18} {'p10':>7} {'p50':>7} {'p90':>7}")
    print("-" * 68)
    for label in sorted(by_label):
        n = len(by_label[label])
        for i, mname in enumerate(METRIC_NAMES):
            deltas = deltas_for(label, mname)
            p10, p50, p90 = np.percentile(deltas, [10, 50, 90])
            lab = label if i == 0 else ""
            ncol = str(n) if i == 0 else ""
            print(f"{lab:<15} {ncol:>4}  Δ{mname:<17} {p10:+7.2f} {p50:+7.2f} {p90:+7.2f}")
        print()

    print("Suggested thresholds (auto-derived from recordings):\n")
    suggested = {}
    for mname in METRIC_NAMES:
        good = [
            d
            for lab, ss in by_label.items() if lab.startswith(good_prefixes)
            for d in (compute_deltas(s, baseline)[mname] for s in ss)
        ]
        bad_groups = {
            lab: [compute_deltas(s, baseline)[mname] for s in ss]
            for lab, ss in by_label.items() if lab.startswith(bad_prefixes)
        }
        if not good or not bad_groups:
            suggested[mname] = getattr(Thresholds(), mname)
            continue
        good_p95 = float(np.percentile(good, 95))
        useful_for = [
            (lab, float(np.percentile(vals, 10)))
            for lab, vals in bad_groups.items()
            if np.percentile(vals, 10) > good_p95
        ]
        if useful_for:
            thr = good_p95 + 0.02
            suggested[mname] = thr
            labs = ", ".join(f"{l}(p10={v:+.2f})" for l, v in useful_for)
            print(f"  {mname:<15} → {thr:+.3f}  catches: {labs}")
        else:
            default = getattr(Thresholds(), mname)
            suggested[mname] = default
            print(f"  {mname:<15} → {default:+.3f}  (kept default; good/bad overlap)")

    th = Thresholds(
        ear_drop=suggested["ear_drop"],
        shoulder_width=suggested["shoulder_width"],
        shoulder_tilt=suggested["shoulder_tilt"],
    )
    if args.write:
        save_thresholds(th)
        print(f"\nWrote thresholds to {THRESHOLDS_PATH}. `monitor` and `visualize` will use them.")
    else:
        print(f"\nDry run. Re-run with --write to save to {THRESHOLDS_PATH}.")


def notify(title: str, body: str):
    subprocess.run(
        ["notify-send", "-a", "posture-nudge", "-u", "normal", title, body],
        check=False,
    )


def log_nudge(reasons: list[str], bad_for_seconds: float):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with NUDGE_LOG.open("a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "reasons": reasons,
            "bad_for_seconds": round(bad_for_seconds, 1),
        }) + "\n")


def log_check(status: str, metrics: Metrics | None = None):
    """Append one check outcome. Status ∈ {good, bad, absent, camera_busy, quiet}.

    Used to build hour-of-day / day-of-week posture heatmaps.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.time(), "status": status}
    if metrics is not None:
        entry["ear_drop"] = round(metrics.ear_drop, 3)
        entry["shoulder_width"] = round(metrics.shoulder_width, 3)
        entry["shoulder_tilt"] = round(metrics.shoulder_tilt, 3)
    with CHECK_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def in_quiet_hours(now: datetime, start: int, end: int) -> bool:
    """True if `now.hour` is inside the quiet window. Handles wrap past midnight."""
    h = now.hour
    if start == end:
        return False
    if start < end:
        return start <= h < end
    return h >= start or h < end


def _capture_burst(pose, cap: cv2.VideoCapture, want: int) -> Metrics | None:
    """Capture up to `want` valid samples from the open camera; return the median."""
    samples: list[Metrics] = []
    max_reads = want * 4  # leave headroom for frames without a valid pose
    reads = 0
    while len(samples) < want and reads < max_reads:
        reads += 1
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            m = compute_metrics(result.pose_landmarks.landmark)
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


def cmd_monitor(args):
    # Both baseline and thresholds are re-read each iteration so `snap-baseline`
    # changes take effect without restarting the service.
    baseline = load_baseline()
    baseline_mtime = BASELINE_PATH.stat().st_mtime
    th = load_thresholds()
    thresholds_mtime = THRESHOLDS_PATH.stat().st_mtime if THRESHOLDS_PATH.exists() else 0.0
    cli_th_overrides = {
        "ear_drop": args.ear_tol,
        "shoulder_width": args.width_tol,
        "shoulder_tilt": args.tilt_tol,
    }

    def apply_overrides(th: Thresholds) -> Thresholds:
        for k, v in cli_th_overrides.items():
            if v is not None:
                setattr(th, k, v)
        return th

    th = apply_overrides(th)

    print(f"Baseline:  ear_drop={baseline.ear_drop:+.2f} sw={baseline.shoulder_width:.2f} "
          f"tilt={baseline.shoulder_tilt:.2f}")
    print(f"Thresh:    ear={th.ear_drop:.2f} width={th.shoulder_width:.2f} "
          f"tilt={th.shoulder_tilt:.2f}")
    print(f"Sampling:  every {args.interval}s, {args.burst_frames} frames per check")
    print(f"Trigger:   {args.required_checks} consecutive bad check(s), cooldown {args.cooldown}s")
    print("Ctrl+C to stop.")

    pose = _new_pose()
    consecutive_bad = 0
    last_notified_at = 0.0

    try:
        while True:
            now_dt = datetime.now()
            stamp = now_dt.strftime("%H:%M:%S")
            if in_quiet_hours(now_dt, args.quiet_start, args.quiet_end):
                log_check("quiet")
                time.sleep(args.interval)
                continue
            # Hot-reload baseline if snap-baseline updated the file.
            try:
                mt = BASELINE_PATH.stat().st_mtime
                if mt != baseline_mtime:
                    baseline = load_baseline()
                    baseline_mtime = mt
                    print(f"[{stamp}] baseline reloaded")
            except FileNotFoundError:
                pass
            # Hot-reload thresholds too (slack slider auto-saves on close).
            try:
                mt = THRESHOLDS_PATH.stat().st_mtime
                if mt != thresholds_mtime:
                    th = apply_overrides(load_thresholds())
                    thresholds_mtime = mt
                    print(f"[{stamp}] thresholds reloaded "
                          f"(ear={th.ear_drop:.2f} width={th.shoulder_width:.2f} "
                          f"tilt={th.shoulder_tilt:.2f})")
            except FileNotFoundError:
                pass
            cap = open_camera(args.camera)
            if cap is None:
                print(f"[{stamp}] camera busy (video call?), skipping this check")
                log_check("camera_busy")
                # Don't let streak tick while the camera is unavailable.
                time.sleep(args.interval)
                continue

            m = _capture_burst(pose, cap, args.burst_frames)
            cap.release()

            if m is None:
                print(f"[{stamp}] no person detected — skipping (resetting streak)")
                log_check("absent")
                consecutive_bad = 0
            else:
                bad, reasons = is_bad_posture(m, baseline, th)
                log_check("bad" if bad else "good", m)
                if bad:
                    consecutive_bad += 1
                    now = time.time()
                    print(f"[{stamp}] BAD ({consecutive_bad}/{args.required_checks}): "
                          f"{'; '.join(reasons)}")
                    if (consecutive_bad >= args.required_checks
                            and now - last_notified_at >= args.cooldown):
                        notify("Sit up straight", "; ".join(reasons))
                        log_nudge(reasons, consecutive_bad * args.interval)
                        last_notified_at = now
                        consecutive_bad = 0  # reset so we don't re-fire on next check
                else:
                    if consecutive_bad > 0:
                        print(f"[{stamp}] OK (streak reset)")
                    consecutive_bad = 0

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pose.close()


def cmd_stats(args):
    if not NUDGE_LOG.exists():
        print("No nudges logged yet.")
        return
    now = time.time()
    cutoff = now - args.days * 86400
    by_day: dict[str, list[dict]] = defaultdict(list)
    reasons_count: dict[str, int] = defaultdict(int)
    total = 0
    for line in NUDGE_LOG.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["ts"] < cutoff:
            continue
        total += 1
        day = datetime.fromtimestamp(e["ts"]).strftime("%Y-%m-%d")
        by_day[day].append(e)
        for r in e["reasons"]:
            key = r.split(" (")[0]
            reasons_count[key] += 1

    print(f"Nudges in last {args.days} day(s): {total}\n")
    if total == 0:
        return
    print(f"{'date':<12} {'count':>5}  {'median bad-for':>15}")
    print("-" * 38)
    for day in sorted(by_day):
        entries = by_day[day]
        med = float(np.median([e["bad_for_seconds"] for e in entries]))
        print(f"{day:<12} {len(entries):>5}  {med:>12.1f}s")

    print("\nTop reasons:")
    for reason, count in sorted(reasons_count.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>4}  {reason}")


def cmd_digest(args):
    """Compute last N days of stats and send a single desktop notification."""
    if not NUDGE_LOG.exists():
        notify("Posture week", "No nudges logged yet — first week!")
        return
    now = time.time()
    cutoff = now - args.days * 86400
    reasons_count: dict[str, int] = defaultdict(int)
    total = 0
    durations: list[float] = []
    for line in NUDGE_LOG.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["ts"] < cutoff:
            continue
        total += 1
        durations.append(e.get("bad_for_seconds", 0.0))
        for r in e["reasons"]:
            reasons_count[r.split(" (")[0]] += 1

    if total == 0:
        notify("Posture week", f"Zero nudges in {args.days} days — either perfect or off-duty.")
        return

    top = sorted(reasons_count.items(), key=lambda kv: -kv[1])[:2]
    top_str = ", ".join(f"{name} ({n})" for name, n in top)
    median_bad = float(np.median(durations)) if durations else 0.0
    body = (f"{total} nudges in {args.days}d. "
            f"Top: {top_str}. Median bad stretch: {median_bad:.0f}s.")
    notify(f"Posture digest", body)
    print(body)


def cmd_plot(args):
    """Render a posture-quality plot to PNG.

    Top:    7-day × 24-hour heatmap of bad-posture fraction (green=good,
            yellow=mixed, red=bad). Cells with no good/bad samples yet are
            shown as muted grey ("no data this hour").
    Bottom: Per-day bar chart of bad-posture % across the same 7 days.
            Days with too few samples (<3 checks) shaded grey.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from datetime import timedelta

    BG = "#0a0a0a"
    FG = "#eceff4"
    MUTED = "#4c566a"
    GREEN = "#a3be8c"
    YELLOW = "#ebcb8b"
    RED = "#bf616a"

    today = datetime.now().date()
    days_back = 7
    day_list = [today - timedelta(days=i) for i in range(days_back - 1, -1, -1)]

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
            if ts.date() not in day_list:
                continue
            counts[day_list.index(ts.date())][ts.hour][s] += 1

    matrix = np.full((days_back, 24), np.nan)
    for di in range(days_back):
        for hi in range(24):
            c = counts[di][hi]
            denom = c["good"] + c["bad"]
            if denom >= 1:
                matrix[di, hi] = c["bad"] / denom

    day_fracs = []
    day_totals = []
    for di in range(days_back):
        good = sum(counts[di][h]["good"] for h in range(24))
        bad = sum(counts[di][h]["bad"] for h in range(24))
        day_totals.append(good + bad)
        day_fracs.append(bad / (good + bad) if (good + bad) > 0 else float("nan"))

    # Render at the size we want it to APPEAR in conky. Conky on hi-DPI X11
    # upscales the PNG by ~2×, so a 300×220 PNG ends up as ~600×440 on screen.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 2.2), facecolor=BG,
                                   gridspec_kw={"height_ratios": [4, 2]}, dpi=100)
    plt.subplots_adjust(left=0.18, right=0.94, top=0.90, bottom=0.18, hspace=1.0)

    cmap = LinearSegmentedColormap.from_list("posture", [GREEN, YELLOW, RED])
    cmap.set_bad(MUTED)

    # Heatmap
    im = ax1.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
                    vmin=0, vmax=1, interpolation="nearest")
    ax1.set_facecolor(MUTED)
    ax1.set_title("Bad-posture % by hour (last 7d)",
                  color=FG, fontsize=10, loc="left", pad=6)
    ax1.set_xticks(range(0, 24, 4))
    ax1.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)],
                        color=FG, fontsize=8)
    ax1.set_yticks(range(days_back))
    ax1.set_yticklabels([d.strftime("%a") for d in day_list],
                        color=FG, fontsize=8)
    for spine in ax1.spines.values():
        spine.set_color(FG)
    ax1.tick_params(colors=FG, length=0, labelsize=8)

    # Per-day bars (bad % per day). Need at least ~1 hour of checks (6 at 10-min
    # interval) to be meaningful — below that the bar is hidden entirely so the
    # day reads as "no data" rather than misleadingly "100% bad on 2 samples".
    MIN_SAMPLES = 6
    bar_colors = []
    bar_heights = []
    for i, f in enumerate(day_fracs):
        if np.isnan(f) or day_totals[i] < MIN_SAMPLES:
            bar_colors.append(MUTED)
            bar_heights.append(0)
        else:
            bar_colors.append(cmap(f))
            bar_heights.append(f * 100)
    ax2.bar(range(days_back), bar_heights, color=bar_colors, width=0.85)
    ax2.set_facecolor(BG)
    ax2.set_title("Bad % per day", color=FG, fontsize=10, loc="left", pad=6)
    ax2.set_xticks(range(days_back))
    ax2.set_xticklabels([d.strftime("%a") for d in day_list],
                        color=FG, fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 50, 100])
    ax2.set_yticklabels(["0", "50", "100"], color=FG, fontsize=8)
    for spine in ax2.spines.values():
        spine.set_color(FG)
    ax2.tick_params(colors=FG, length=0, labelsize=8)
    ax2.grid(True, axis="y", alpha=0.18, color=FG)
    ax2.set_axisbelow(True)

    fig.savefig(args.output, dpi=100, facecolor=BG)
    plt.close(fig)
    total = sum(day_totals)
    print(f"wrote {args.output} ({total} good/bad samples across {days_back} days)")

SYSTEMD_UNIT = """[Unit]
Description=Posture nudge — webcam posture monitor
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart={python} {script} monitor --interval 600 --required-checks 1 --cooldown 1800 --quiet-start 22 --quiet-end 7
Restart=on-failure
RestartSec=15
# Environment needed for notify-send + webcam access
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=%t

[Install]
WantedBy=graphical-session.target
"""


def cmd_install_service(args):
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "posture-nudge.service"
    # Don't resolve python — would follow the venv symlink to system python (missing deps).
    python_bin = Path(sys.executable)
    script_path = Path(__file__).resolve()
    unit_path.write_text(SYSTEMD_UNIT.format(python=python_bin, script=script_path))
    print(f"Wrote {unit_path}\n")
    print("To enable and start:")
    print("  systemctl --user daemon-reload")
    print("  systemctl --user enable --now posture-nudge.service")
    print("\nTo check status / logs:")
    print("  systemctl --user status posture-nudge.service")
    print("  journalctl --user -u posture-nudge.service -f")
    print("\nTo stop / disable:")
    print("  systemctl --user disable --now posture-nudge.service")


DIGEST_SERVICE = """[Unit]
Description=Posture nudge — weekly stats digest

[Service]
Type=oneshot
ExecStart={python} {script} digest --days 7
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=%t
"""

DIGEST_TIMER = """[Unit]
Description=Run posture-nudge weekly digest every Sunday at 20:00

[Timer]
OnCalendar=Sun *-*-* 20:00:00
Persistent=true
Unit=posture-nudge-digest.service

[Install]
WantedBy=timers.target
"""


PLOT_SERVICE = """[Unit]
Description=Posture nudge — regenerate plot.png

[Service]
Type=oneshot
ExecStart={python} {script} plot
"""

PLOT_TIMER = """[Unit]
Description=Regenerate posture-nudge plot every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Unit=posture-nudge-plot.service

[Install]
WantedBy=timers.target
"""


def cmd_install_plot_timer(args):
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    python_bin = Path(sys.executable)
    script_path = Path(__file__).resolve()

    svc_path = unit_dir / "posture-nudge-plot.service"
    tim_path = unit_dir / "posture-nudge-plot.timer"
    svc_path.write_text(PLOT_SERVICE.format(python=python_bin, script=script_path))
    tim_path.write_text(PLOT_TIMER)

    print(f"Wrote {svc_path}")
    print(f"Wrote {tim_path}\n")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now",
                    "posture-nudge-plot.timer"], check=True)
    print("Plot is regenerated every 5 minutes from now on.")
    print("Disable: systemctl --user disable --now posture-nudge-plot.timer")


def cmd_install_digest(args):
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    python_bin = Path(sys.executable)
    script_path = Path(__file__).resolve()

    svc_path = unit_dir / "posture-nudge-digest.service"
    tim_path = unit_dir / "posture-nudge-digest.timer"
    svc_path.write_text(DIGEST_SERVICE.format(python=python_bin, script=script_path))
    tim_path.write_text(DIGEST_TIMER)

    print(f"Wrote {svc_path}")
    print(f"Wrote {tim_path}\n")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now",
                    "posture-nudge-digest.timer"], check=True)
    result = subprocess.run(
        ["systemctl", "--user", "list-timers", "posture-nudge-digest.timer",
         "--no-pager"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    print("Test right now: systemctl --user start posture-nudge-digest.service")
    print("Disable:        systemctl --user disable --now posture-nudge-digest.timer")


def main():
    p = argparse.ArgumentParser(description="Posture nudge — webcam posture monitor")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate")
    pc.add_argument("--duration", type=float, default=5.0)
    pc.add_argument("--camera", type=int, default=0)
    pc.set_defaults(func=cmd_calibrate)

    pv = sub.add_parser("visualize")
    pv.add_argument("--camera", type=int, default=0)
    pv.set_defaults(func=cmd_visualize)

    pr = sub.add_parser("record", help="Capture labeled posture samples")
    pr.add_argument("--camera", type=int, default=0)
    pr.add_argument("--duration", type=float, default=4.0, help="Capture seconds per posture")
    pr.add_argument("--prep", type=float, default=3.0, help="Countdown seconds before each capture")
    pr.add_argument("--fresh", action="store_true", help="Wipe existing samples before recording")
    pr.add_argument("--label", type=str, default=None,
                    help="Only record for this single label (otherwise runs full sequence)")
    pr.set_defaults(func=cmd_record)

    pa = sub.add_parser("analyze", help="Summarize samples and pick thresholds")
    pa.add_argument("--write", action="store_true",
                    help=f"Save suggested thresholds to {THRESHOLDS_PATH}")
    pa.add_argument("--refit-baseline", action="store_true",
                    help="Replace baseline.json with the median of good_* samples")
    pa.set_defaults(func=cmd_analyze)

    pm = sub.add_parser("monitor", help="Periodic posture check with desktop alerts")
    pm.add_argument("--camera", type=int, default=0)
    pm.add_argument("--interval", type=float, default=120.0,
                    help="Seconds between checks (camera is released between checks)")
    pm.add_argument("--burst-frames", type=int, default=15,
                    help="Frames to capture per check (median-aggregated)")
    pm.add_argument("--required-checks", type=int, default=2,
                    help="Consecutive bad checks required before notifying")
    pm.add_argument("--cooldown", type=float, default=300.0,
                    help="Seconds between notifications")
    pm.add_argument("--ear-tol", type=float, default=None)
    pm.add_argument("--width-tol", type=float, default=None)
    pm.add_argument("--tilt-tol", type=float, default=None)
    pm.add_argument("--quiet-start", type=int, default=22,
                    help="Hour (0-23) when quiet window starts; no nudges between start/end")
    pm.add_argument("--quiet-end", type=int, default=7,
                    help="Hour (0-23) when quiet window ends")
    pm.set_defaults(func=cmd_monitor)

    ps = sub.add_parser("stats", help="Summarize recent nudges")
    ps.add_argument("--days", type=int, default=7)
    ps.set_defaults(func=cmd_stats)

    pi = sub.add_parser("install-service", help="Install a systemd user unit for monitor")
    pi.set_defaults(func=cmd_install_service)

    pd = sub.add_parser("digest", help="Send a weekly stats summary as a desktop notification")
    pd.add_argument("--days", type=int, default=7)
    pd.set_defaults(func=cmd_digest)

    pid = sub.add_parser("install-digest",
                         help="Install a systemd user timer that runs `digest` weekly")
    pid.set_defaults(func=cmd_install_digest)

    pp = sub.add_parser("plot",
                        help="Render a posture-quality PNG (heatmap + daily bars)")
    pp.add_argument("--output", type=str, default=str(PLOT_PATH))
    pp.set_defaults(func=cmd_plot)

    ppt = sub.add_parser("install-plot-timer",
                         help="systemd user timer that regenerates plot.png every 5min")
    ppt.set_defaults(func=cmd_install_plot_timer)

    psb = sub.add_parser("snap-baseline",
                         help="Quick re-calibrate the baseline (for chair/screen changes)")
    psb.add_argument("--camera", type=int, default=0)
    psb.add_argument("--duration", type=float, default=5.0)
    psb.add_argument("--test-seconds", type=float, default=20.0,
                     help="After saving, run a live test phase so you can verify "
                          "the new baseline by slouching/leaning. 0 to skip.")
    psb.add_argument("--no-gui", action="store_true",
                     help="Skip the live preview window (headless mode)")
    psb.set_defaults(func=cmd_snap_baseline)

    pk = sub.add_parser("install-shortcut",
                        help="Bind a GNOME keyboard shortcut to snap-baseline")
    pk.add_argument("--binding", default="<Control><Alt><Shift>p",
                    help="GNOME shortcut, e.g. '<Control><Alt><Shift>p'")
    pk.set_defaults(func=cmd_install_shortcut)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
