# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['Keil2Cmake.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'keil2cmake_cli',
        'keil2cmake_common',
        'i18n',
        'project_gen',
        'keil.uvprojx',
        'keil.device',
        'keil.config',
        'compiler.toolchains',
        'compiler.presets',
        'compiler.clangd',
        'compiler.templates',
        'compiler.armgcc.layout',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'numpy', 'pandas'],
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
    name='Keil2Cmake',
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
