# PyInstaller spec — single-file build. Run via: pyinstaller posture_app.spec
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

mp_data = collect_data_files("mediapipe")
hidden = collect_submodules("mediapipe.python") + [
    "PIL.Image", "PIL.ImageTk",
]

datas = mp_data + [
    ("chair.png", "."),
]

a = Analysis(
    ["posture_app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "test", "unittest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="PostureNudge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
