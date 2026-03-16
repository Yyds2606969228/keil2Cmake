# -*- coding: utf-8 -*-

import argparse
import os
import sys

from .keil.config import get_language
from .keil.uvprojx import parse_uvprojx
from .keil.sync import sync_cmake_to_uvprojx
from .keil.device import detect_cpu_architecture
from .project_gen import generate_cmake_structure
from .compiler.toolchains import generate_toolchains
from .compiler.presets import generate_cmake_presets
from .compiler.clangd import generate_clangd_config
from .compiler.debug import generate_debug_templates, generate_openocd_files
from .i18n import set_language, t


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Keil uVision to CMake converter (ARM-GCC only, with clangd support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  %(prog)s project.uvprojx -o ./cmake_project
        '''
    )

    parser.add_argument('uvprojx', help='Path to Keil .uvprojx project file')
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Output directory for generated CMake project',
    )
    return parser


def build_openocd_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate OpenOCD/cortex-debug templates for existing CMake projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  %(prog)s -mcu STM32F103C8 -debugger jlink
        '''
    )
    parser.add_argument('-mcu', required=True, help='MCU device name, e.g. STM32F103C8')
    parser.add_argument(
        '-debugger',
        required=True,
        choices=['daplink', 'jlink', 'stlink'],
        help='Debugger probe: daplink/jlink/stlink',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing openocd.cfg/launch.json/tasks.json',
    )
    return parser


def build_sync_keil_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Sync CMake user config back to Keil .uvprojx',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  %(prog)s --uvprojx ./project.uvprojx --cmake-root ./cmake_project
  %(prog)s --uvprojx ./project.uvprojx --prune
        '''
    )
    parser.add_argument('--uvprojx', required=True, help='Target .uvprojx path to update')
    parser.add_argument(
        '--cmake-root',
        default='.',
        help='CMake project root containing cmake/user/keil2cmake_user.cmake',
    )
    parser.add_argument('--target', help='TargetName inside uvprojx (default: first target)')
    parser.add_argument('--group', default='K2C_Sync', help='Keil group used for synchronized sources')
    parser.add_argument('--prune', action='store_true', help='Prune stale files only inside sync group')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without writing uvprojx')
    return parser


def _main_convert(argv) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_language(get_language())

    if not os.path.exists(args.uvprojx):
        print(t('cli.error.file_not_found', path=args.uvprojx))
        return 1

    project_data = parse_uvprojx(args.uvprojx)
    project_root = os.path.abspath(args.output)
    os.makedirs(project_root, exist_ok=True)

    generate_cmake_structure(project_data, project_root)
    generate_toolchains(project_data, project_root)
    generate_cmake_presets(project_root, project_data)
    generate_clangd_config(
        project_root,
        detect_cpu_architecture(project_data['device']),
        project_data.get('use_microlib'),
    )
    generate_debug_templates(project_root)
    generate_openocd_files(
        project_root,
        project_data.get('device', ''),
        project_data.get('debugger', ''),
        overwrite=False,
    )

    print('\n' + t('cli.done'))
    print(f"  {t('cli.summary.project')}: {project_data['project_name']}")
    print(f"  {t('cli.summary.device')}: {project_data['device']}")
    print(f"  {t('cli.summary.keil_compiler')}: {project_data.get('keil_compiler', 'armcc')}")
    print(f"  {t('cli.summary.compiler')}: armgcc")
    print(f"  {t('cli.summary.optimize')}: -O{project_data['opt_level']}")
    print(f"  {t('cli.summary.output')}: {os.path.abspath(project_root)}")
    if project_data.get('use_microlib') is not None:
        microlib_value = 'ON' if project_data.get('use_microlib') else 'OFF'
        print(f"  {t('cli.summary.microlib')}: {microlib_value}")
    if project_data.get('asm_detected'):
        print(t('cli.compat.armasm'))

    print('\n' + t('cli.build_cmds'))
    print('  cmake --preset keil2cmake')
    print('  cmake --build --preset build')
    print('  cmake --build --preset check')
    print(t('cli.compat.note'))
    return 0


def _main_openocd(argv) -> int:
    parser = build_openocd_parser()
    args = parser.parse_args(argv)

    set_language(get_language())
    generate_debug_templates(os.getcwd())
    result = generate_openocd_files(os.getcwd(), args.mcu, args.debugger, overwrite=args.overwrite)

    print('\n' + t('cli.openocd.done'))
    print(f"  {t('cli.summary.device')}: {args.mcu}")
    print(f"  {t('cli.openocd.summary.debugger')}: {args.debugger}")
    print(f"  {t('cli.openocd.summary.output')}: {os.path.abspath(os.getcwd())}")
    print(f"  {t('cli.openocd.summary.openocd_path')}: {result['openocd_path']}")
    print(f"  {t('cli.openocd.summary.gdb_path')}: {result['gdb_path']}")
    if result.get('openocd_interface'):
        print(f"  {t('cli.openocd.summary.interface')}: {result['openocd_interface']}")
    if result.get('openocd_target'):
        print(f"  {t('cli.openocd.summary.target')}: {result['openocd_target']}")
    print(f"  {t('cli.openocd.summary.executable')}: {result['debug_executable']}")
    print(f"  {t('cli.openocd.summary.launch')}: {result['launch_json']}")
    print(f"  {t('cli.openocd.summary.tasks')}: {result['tasks_json']}")
    print(f"  {t('cli.openocd.summary.config')}: {result['openocd_cfg']}")
    return 0


def _main_sync_keil(argv) -> int:
    parser = build_sync_keil_parser()
    args = parser.parse_args(argv)

    set_language(get_language())

    try:
        result = sync_cmake_to_uvprojx(
            uvprojx_path=args.uvprojx,
            cmake_root=args.cmake_root,
            target_name=args.target,
            group_name=args.group,
            prune=args.prune,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f'Error: {exc}')
        return 1

    action = 'Dry run complete' if args.dry_run else 'Sync complete'
    print('\n' + action)
    print(f'  uvprojx: {result.uvprojx_path}')
    print(f'  target: {result.target_name}')
    print(f'  group: {result.group_name}')
    print(f'  sources listed: {result.source_count}')
    print(f'  files added: {result.added_count}')
    print(f'  files removed: {result.removed_count}')
    print(f'  include dirs: {result.include_count}')
    print(f'  defines: {result.define_count}')
    if not args.dry_run:
        print(f'  backup: {result.backup_path}')
    return 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == 'openocd':
        return _main_openocd(argv[1:])
    if argv and argv[0] == 'sync-keil':
        return _main_sync_keil(argv[1:])
    return _main_convert(argv)
