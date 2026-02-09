# -*- coding: utf-8 -*-

import argparse
import os
import sys

from .keil.config import get_language
from .keil.uvprojx import parse_uvprojx
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
    result = generate_openocd_files(os.getcwd(), args.mcu, args.debugger)

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


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == 'openocd':
        return _main_openocd(argv[1:])
    return _main_convert(argv)
