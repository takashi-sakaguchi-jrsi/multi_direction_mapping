# -*- mode: python ; coding: utf-8 -*-
"""3方向合成製品版 PyInstaller spec。"""

import os

project_root = os.path.abspath(SPECPATH)
config_file = os.path.join(project_root, "data", "config", "default_config.json")

validation_hidden = [
    "src.validation",
    "src.validation.best_view_accumulator",
    "src.validation.frame_analyzer",
    "src.validation.geometry",
    "src.validation.known_z",
    "src.validation.map_registration",
    "src.validation.ocr_simulation",
    "src.validation.records",
    "src.validation.report",
    "src.validation.seam_join",
    "src.validation.seam_warp",
    "src.validation.sideview_feature_matcher",
    "src.validation.sideview_projection_mapper",
    "src.validation.strip_correction",
    "src.validation.fisheye_sideview_renderer",
    "src.validation.pose_jitter",
    "src.product_runtime",
    "src.main_two_direction_validation",
]

a = Analysis(
    ["src/main_twopass.py"],
    pathex=[
        project_root,
        os.path.join(project_root, "src"),
    ],
    binaries=[],
    datas=[
        (config_file, "data/config"),
    ],
    hiddenimports=[
        "src.config",
        "src.ocr_utils",
        "src.coordinate_transform",
        "src.debug_visualizer",
        "src.feature_matching",
        "src.feature_matching_cylindrical",
        "src.camera_estimation",
        "src.vanishing_point_estimator",
        "src.colormap_generator",
        "src.colormap_correction",
        "src.progress_reporter",
        "src.camera_utils",
        "src.auto_tuner",
        "src.calibration",
        "src.image_size_correction",
        "cv2",
        "numpy",
        "pandas",
        "scipy",
        "scipy.ndimage",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "openpyxl",
        "pytesseract",
        "PIL",
        "src",
        *validation_hidden,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="main_twopass",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="main_twopass",
)
