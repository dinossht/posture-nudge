#!/usr/bin/env python3
"""One-shot character popup for posture-nudge.

Uses GTK3 + ARGB visual so the chair PNG's transparent background actually
shows through (tkinter on X11 can't do per-pixel alpha). Slides up from the
bottom-right corner, holds, then slides back down and exits.

Run with the system python (which has python3-gi); the venv doesn't.
"""

import argparse
import sys
from pathlib import Path

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib  # noqa: E402

CHAIR_IMG = Path(__file__).resolve().parent / "chair.png"

MARGIN_RIGHT = 0
MARGIN_BOTTOM = 0
ANIM_STEP_PX = 14
ANIM_TICK_MS = 14


class Popup(Gtk.Window):
    def __init__(self, hold_ms: int):
        super().__init__(type=Gtk.WindowType.POPUP)
        self.set_decorated(False)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_keep_above(True)
        self.set_accept_focus(False)
        self.set_app_paintable(True)
        self.set_resizable(False)

        # ARGB visual so transparent PNG pixels are actually transparent.
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)

        if not CHAIR_IMG.exists():
            print(f"chair image not found: {CHAIR_IMG}", file=sys.stderr)
            sys.exit(1)

        pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(CHAIR_IMG))
        image = Gtk.Image.new_from_pixbuf(pixbuf)
        self.add(image)

        # Use the screen geometry of the primary monitor.
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor() or display.get_monitor(0)
        geom = monitor.get_geometry()
        self.target_x = geom.x + geom.width - pixbuf.get_width() - MARGIN_RIGHT
        self.target_y = geom.y + geom.height - pixbuf.get_height() - MARGIN_BOTTOM
        self.start_y = geom.y + geom.height + 10  # off-screen below
        self.current_y = self.start_y

        self.move(self.target_x, self.start_y)
        self.connect("realize", self._on_realize)
        self.connect("draw", self._on_draw)

        self._hold_ms = hold_ms

    def _on_realize(self, _w):
        # Schedule slide-in on first realize so the move() above is honored.
        GLib.timeout_add(ANIM_TICK_MS, self._slide_in)

    def _on_draw(self, _w, cr):
        # Clear the window's background so non-image regions are fully transparent.
        cr.set_source_rgba(0, 0, 0, 0)
        cr.set_operator(1)  # CAIRO_OPERATOR_SOURCE
        cr.paint()
        return False  # let children draw

    def _slide_in(self):
        if self.current_y > self.target_y:
            self.current_y = max(self.target_y, self.current_y - ANIM_STEP_PX)
            self.move(self.target_x, self.current_y)
            return True
        # Settled — schedule slide-out.
        GLib.timeout_add(self._hold_ms, self._begin_slide_out)
        return False

    def _begin_slide_out(self):
        GLib.timeout_add(ANIM_TICK_MS, self._slide_out)
        return False

    def _slide_out(self):
        if self.current_y < self.start_y:
            self.current_y = min(self.start_y, self.current_y + ANIM_STEP_PX)
            self.move(self.target_x, self.current_y)
            return True
        Gtk.main_quit()
        return False


def main():
    p = argparse.ArgumentParser()
    # Keep the args compatible with the previous tkinter version even though
    # we now ignore them — the chair image carries the meaning on its own.
    p.add_argument("title", nargs="?", default="")
    p.add_argument("body", nargs="?", default="")
    p.add_argument("--duration-ms", type=int, default=4500)
    args = p.parse_args()

    win = Popup(hold_ms=args.duration_ms)
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
