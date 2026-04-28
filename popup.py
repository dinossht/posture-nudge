#!/usr/bin/env python3
"""One-shot posture-nudge character popup.

Spawned by the monitor each time a nudge fires. Slides up from the
bottom-right corner with an emoji "character" + the reason, lingers a few
seconds, then slides back down and exits. Pure stdlib (tkinter) so it stays
light — no mediapipe / opencv import, fires in <100ms.
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path

BG = "#0a0a0a"
FG = "#eceff4"
ACCENT = "#bf616a"
SUBTLE = "#8fbcbb"

CHAIR_IMG = Path(__file__).resolve().parent / "chair.png"

WIDTH = 320
HEIGHT = 110
MARGIN_RIGHT = 32
MARGIN_BOTTOM = 60
ANIM_STEP_PX = 12   # pixels per animation tick
ANIM_TICK_MS = 12   # ms between animation ticks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("title", default="Sit up!", nargs="?")
    p.add_argument("body", default="head forward", nargs="?")
    p.add_argument("--emoji", default="🪑")
    p.add_argument("--duration-ms", type=int, default=4500)
    args = p.parse_args()

    root = tk.Tk()
    root.overrideredirect(True)            # no titlebar / borders
    root.wm_attributes("-topmost", True)
    root.wm_attributes("-type", "splash")  # hint to WMs (KDE/sway) to skip taskbar
    root.configure(bg=BG)

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    final_x = sw - WIDTH - MARGIN_RIGHT
    final_y = sh - HEIGHT - MARGIN_BOTTOM
    start_y = sh + 5  # just below the screen

    root.geometry(f"{WIDTH}x{HEIGHT}+{final_x}+{start_y}")

    # Subtle accent bar on the left edge
    accent = tk.Frame(root, bg=ACCENT, width=4)
    accent.pack(side="left", fill="y")

    body = tk.Frame(root, bg=BG, padx=14, pady=10)
    body.pack(side="left", fill="both", expand=True)

    if CHAIR_IMG.exists():
        chair_img = tk.PhotoImage(file=str(CHAIR_IMG))
        char_label = tk.Label(body, image=chair_img, bg=BG, borderwidth=0)
        char_label.image = chair_img  # keep reference so GC doesn't free it
    else:
        char_label = tk.Label(body, text=args.emoji, bg=BG, fg=FG,
                              font=("Sans", 36))
    char_label.pack(side="left", padx=(0, 14))

    text_frame = tk.Frame(body, bg=BG)
    text_frame.pack(side="left", anchor="center")

    title = tk.Label(text_frame, text=args.title, bg=BG, fg=FG,
                     font=("Sans", 14, "bold"), anchor="w", justify="left")
    title.pack(anchor="w")
    sub = tk.Label(text_frame, text=args.body, bg=BG, fg=SUBTLE,
                   font=("Sans", 11), anchor="w", justify="left",
                   wraplength=WIDTH - 110)
    sub.pack(anchor="w")

    state = {"phase": "in", "y": start_y}

    def tick():
        y = state["y"]
        if state["phase"] == "in":
            if y <= final_y:
                state["y"] = final_y
                root.geometry(f"{WIDTH}x{HEIGHT}+{final_x}+{final_y}")
                root.after(args.duration_ms, lambda: state.update(phase="out"))
                root.after(ANIM_TICK_MS, tick)
                return
            state["y"] = max(final_y, y - ANIM_STEP_PX)
        elif state["phase"] == "out":
            if y >= start_y:
                root.destroy()
                return
            state["y"] = min(start_y, y + ANIM_STEP_PX)
        else:
            return
        root.geometry(f"{WIDTH}x{HEIGHT}+{final_x}+{state['y']}")
        root.after(ANIM_TICK_MS, tick)

    root.after(20, tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
