# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

a = Analysis(
    ['src/openocd_mcp/scripts/openocd_mcp.py'],
    pathex=['src/openocd_mcp/src'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'openocd_mcp',
        'openocd_mcp.server',
        'openocd_mcp.tools.service',
        'openocd_mcp.transport.openocd_tcl.client',
        'openocd_mcp.transport.serial.manager',
        'openocd_mcp.transport.probe_discovery',
        'openocd_mcp.parsers.svd_resolver',
        'openocd_mcp.parsers.elf_resolver',
        'openocd_mcp.runtime.task_runtime',
        'openocd_mcp.runtime.context',
        'openocd_mcp.runtime.events',
        'openocd_mcp.core.errors',
        'openocd_mcp.core.models',
        'openocd_mcp.core.session',
        'openocd_mcp.core.openocd_path',
        'fastmcp',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pandas'],
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
    name='openocd-mcp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
