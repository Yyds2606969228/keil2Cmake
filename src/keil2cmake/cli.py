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
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing openocd.cfg/launch.json/tasks.json',
    )
    return parser


def build_onnx_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate TinyML artifacts from ONNX models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  %(prog)s --model model.onnx --weights flash --emit c
        '''
    )
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument(
        '--weights',
        default='flash',
        choices=['flash', 'ram'],
        help='Weight storage location',
    )
    parser.add_argument(
        '--emit',
        default='c',
        choices=['c', 'lib'],
        help='Emit C source or static library',
    )
    parser.add_argument(
        '--output',
        default='./onnx-for-mcu',
        help='Output root directory',
    )
    parser.add_argument(
        '--no-strict-validation',
        dest='strict_validation',
        action='store_false',
        help='Allow generation when consistency validation is skipped.',
    )
    parser.set_defaults(strict_validation=True)
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


def _main_onnx(argv) -> int:
    parser = build_onnx_parser()
    args = parser.parse_args(argv)

    set_language(get_language())
    # Lazy import keeps non-TinyML commands/help free from heavy runtime deps.
    from .tinyml import generate_tinyml_project

    if not os.path.exists(args.model):
        print(t('cli.error.file_not_found', path=args.model))
        return 1

    result = generate_tinyml_project(
        args.model,
        args.output,
        args.weights,
        args.emit,
        strict_validation=args.strict_validation,
    )

    print('\n' + t('cli.onnx.done'))
    print(f"  {t('cli.onnx.summary.model')}: {result['model_name']}")
    print(f"  {t('cli.onnx.summary.output')}: {os.path.abspath(result['project_dir'])}")
    print(f"  {t('cli.onnx.summary.backend')}: {result['backend']}")
    print(f"  {t('cli.onnx.summary.weights')}: {result['weights']}")
    print(f"  {t('cli.onnx.summary.emit')}: {args.emit}")
    print(f"  {t('cli.onnx.summary.header')}: {result['header']}")
    print(f"  {t('cli.onnx.summary.source')}: {result['source']}")
    print(f"  {t('cli.onnx.summary.manifest')}: {result['manifest']}")
    if result.get('library'):
        print(f"  {t('cli.onnx.summary.library')}: {result['library']}")
    validation = result.get("validation")
    if validation is not None:
        status_map = {
            "passed": t("cli.onnx.validation.passed"),
            "skipped": t("cli.onnx.validation.skipped"),
            "failed": t("cli.onnx.validation.failed"),
        }
        status_label = status_map.get(getattr(validation, "status", ""), str(getattr(validation, "status", "")))
        detail = getattr(validation, "reason", "") or ""
        engine = getattr(validation, "engine", "") or ""
        extras = []
        if engine:
            extras.append(engine)
        if detail:
            extras.append(detail)
        if extras:
            print(f"  {t('cli.onnx.summary.validation')}: {status_label} ({'; '.join(extras)})")
        else:
            print(f"  {t('cli.onnx.summary.validation')}: {status_label}")
    return 0


def _main_mcp_debug(argv) -> int:
    if argv:
        print('mcp-debug does not accept extra arguments.')
        return 2

    # Lazy import keeps the default conversion and TinyML paths independent
    # from the optional debug runtime dependencies.
    from openocd_mcp.server import main as mcp_main

    mcp_main()
    return 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == 'openocd':
        return _main_openocd(argv[1:])
    if argv and argv[0] == 'mcp-debug':
        return _main_mcp_debug(argv[1:])
    if argv and argv[0] == 'onnx':
        return _main_onnx(argv[1:])
    return _main_convert(argv)
