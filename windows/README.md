# Posture Nudge — Windows

Single-file desktop app that watches your posture through the webcam and pops
up a friendly chair character when you slouch.

## For users (your colleague)

1. Download `PostureNudge.exe` (one file, ~250 MB — heavy because the AI model
   is bundled).
2. Double-click to run. Windows SmartScreen may warn about an unrecognized
   publisher — click **More info → Run anyway**.
3. First run: click **Calibrate / Test**. Sit upright for 5 seconds, then
   you'll get a 20-second test phase where you can try slouching to see how
   it reacts. Drag the **slack** slider if it feels too strict; the value is
   saved when the dialog closes.
4. Click **Start monitoring**. The app will check posture every 10 minutes
   between 07:00 and 18:00 (configurable on the Settings tab) and show a
   slide-up chair popup when posture is bad.
5. Optionally tick **Start with Windows** in Settings so it runs automatically
   on login.

The **Stats** tab shows a 7-day × 12-hour heatmap of bad-posture fraction
plus a daily summary bar chart. Data is logged to `%APPDATA%\PostureNudge\`.

## For maintainers (building the .exe)

The .exe is built by GitHub Actions on every push to `main`. To trigger a
build manually: **Actions → Build Windows .exe → Run workflow**. The
artifact `PostureNudge-windows` shows up under the workflow run.

To build locally on a Windows machine:
```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller==6.10.0
pyinstaller --clean -y posture_app.spec
# Output: dist\PostureNudge.exe
```

## Notes

- This is a fork of the Linux project — it does not share the systemd timer /
  conky widget machinery; everything lives in one Tk window.
- Data files (`baseline.json`, `thresholds.json`, `check_log.jsonl`,
  `nudge_log.jsonl`, `settings.json`) live in `%APPDATA%\PostureNudge\` so
  uninstalling the .exe doesn't trash history.
