# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ["scripts/K2CTinyML.py"],
    pathex=["src"],
    binaries=[],
    datas=[("src/k2c_tinyml/templates", "k2c_tinyml/templates")],
    hiddenimports=[
        "k2c_tinyml.cli",
        "k2c_tinyml.converter",
        "k2c_tinyml.runtime",
        "k2c_tinyml.backends",
        "k2c_tinyml.backends.c.backend",
        "k2c_tinyml.backends.c.ops",
        "k2c_tinyml.backends.c.ops.registry",
        "jinja2",
        "onnx",
        "numpy",
        "onnxruntime",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pandas"],
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
    name="K2CTinyML",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

