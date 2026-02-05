# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import Tree

block_cipher = None

a = Analysis(
    ['scripts/Keil2Cmake.py'],
    pathex=['src'],
    binaries=[],
    datas=[Tree('src/keil2cmake/templates', prefix='keil2cmake/templates')],
    hiddenimports=[
        'keil2cmake.cli',
        'keil2cmake.common',
        'keil2cmake.i18n',
        'keil2cmake.project_gen',
        'keil2cmake.template_engine',
        'keil2cmake.keil.uvprojx',
        'keil2cmake.keil.device',
        'keil2cmake.keil.config',
        'keil2cmake.compiler.toolchains',
        'keil2cmake.compiler.presets',
        'keil2cmake.compiler.clangd',
        'keil2cmake.compiler.armgcc.layout',
        'jinja2',
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
