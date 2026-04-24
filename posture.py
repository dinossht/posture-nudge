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
    required = [L_EAR, R_EAR, L_SHOULDER, R_SHOULDER]
    confidence = min(lm[i].visibility for i in required)
    if confidence < 0.5:
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


def open_camera(index: int) -> cv2.VideoCapture | None:
    """Open camera; returns None if unavailable (so callers can retry)."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


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

    Intended to be triggered by a keyboard shortcut after the user has adjusted
    their chair, screen tilt, or desk setup. Headless (notification-based feedback).
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

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

    pose = _new_pose()
    notify("posture-nudge", f"Hold your good posture — capturing {args.duration:.0f}s")
    time.sleep(0.8)  # let user read the notification

    samples: list[Metrics] = []
    end = time.time() + args.duration
    try:
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                continue
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                m = compute_metrics(result.pose_landmarks.landmark)
                if m is not None:
                    samples.append(m)
    finally:
        cap.release()
        pose.close()

    if len(samples) < 10:
        notify("posture-nudge", f"Snap failed — only {len(samples)} samples. Sit upright and retry.")
        sys.exit(1)

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
    # baseline is re-read each iteration so `snap-baseline` takes effect without restart.
    baseline = load_baseline()
    baseline_mtime = BASELINE_PATH.stat().st_mtime
    th = load_thresholds()
    if args.ear_tol is not None:
        th.ear_drop = args.ear_tol
    if args.width_tol is not None:
        th.shoulder_width = args.width_tol
    if args.tilt_tol is not None:
        th.shoulder_tilt = args.tilt_tol

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
            cap = open_camera(args.camera)
            if cap is None:
                print(f"[{stamp}] camera busy (video call?), skipping this check")
                # Don't let streak tick while the camera is unavailable.
                time.sleep(args.interval)
                continue

            m = _capture_burst(pose, cap, args.burst_frames)
            cap.release()

            if m is None:
                print(f"[{stamp}] no person detected — skipping (resetting streak)")
                consecutive_bad = 0
            else:
                bad, reasons = is_bad_posture(m, baseline, th)
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

    psb = sub.add_parser("snap-baseline",
                         help="Quick re-calibrate the baseline (for chair/screen changes)")
    psb.add_argument("--camera", type=int, default=0)
    psb.add_argument("--duration", type=float, default=3.0)
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
