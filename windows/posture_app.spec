# PyInstaller spec — single-file build. Run via: pyinstaller posture_app.spec
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

mp_data = collect_data_files("mediapipe")
mpl_data = collect_data_files("matplotlib")
hidden = (
    collect_submodules("mediapipe.python")
    + collect_submodules("matplotlib")
    + ["PIL.Image", "PIL.ImageTk", "matplotlib.backends.backend_tkagg"]
)

datas = mp_data + mpl_data + [
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
    # No excludes — earlier 'unittest' exclude broke matplotlib at runtime
    # (matplotlib/__init__.py line 159 imports it).
    excludes=[],
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
