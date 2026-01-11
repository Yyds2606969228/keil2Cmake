# -*- coding: utf-8 -*-

import argparse
import os

from keil2cmake_common import SUPPORTED_COMPILERS
from keil.config import load_config, edit_config, get_language
from keil.uvprojx import parse_uvprojx
from keil.device import detect_cpu_architecture
from project_gen import generate_cmake_structure, clean_generated
from compiler.toolchains import generate_toolchains
 
from compiler.presets import generate_cmake_presets
from compiler.clangd import generate_clangd_config
from i18n import set_language, t


def compute_project_root(args_output: str | None, uvprojx_path: str) -> str:
    # Default output root is derived from the uvprojx location.
    # Backward compatible: when uvprojx is provided, `-o .` means auto-derived as well.
    if (not args_output) or str(args_output).strip() == '.':
        uvprojx_dir = os.path.dirname(os.path.abspath(uvprojx_path))
        uvprojx_dir_name = os.path.basename(uvprojx_dir).lower()
        if uvprojx_dir_name in ('mdk-arm', 'mdk_arm', 'mdkarm'):
            return os.path.dirname(uvprojx_dir)
        return uvprojx_dir

    return os.path.abspath(args_output)


def build_parser() -> argparse.ArgumentParser:
    # Note: We can't use t() here yet because language isn't initialized.
    # The parser is created before args are parsed, so we use a mix of English
    # and will show localized help in error messages.
    parser = argparse.ArgumentParser(
        description='Keil uVision to CMake converter for ARM Embedded Toolchains (with clangd support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  %(prog)s project.uvprojx                    # Convert Keil project to CMake
  %(prog)s -e ARMCC_PATH=D:/Keil/ARMCC/bin/  # Edit compiler path
  %(prog)s -e ARMCC_INCLUDE=D:/Keil/include/ # Edit include path  
  %(prog)s --show-config                      # Show current configuration
  %(prog)s -o ./build project.uvprojx         # Specify output directory
  %(prog)s --clean -o .                       # Clean generated files
  %(prog)s --lang zh project.uvprojx          # Use Chinese output
        '''
    )

    parser.add_argument('uvprojx', nargs='?', help='Path to Keil .uvprojx project file')
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='Output directory. Default: auto-derived from .uvprojx (MDK-ARM → parent)',
    )
    parser.add_argument('--clean', action='store_true', help='Clean generated CMake files')
    parser.add_argument('--lang', choices=['zh', 'en'], default=None, help='Language: zh (Chinese) or en (English)')
    parser.add_argument('--compiler', choices=list(SUPPORTED_COMPILERS), default=None,
                        help='Override compiler: armcc / armclang / armgcc')
    parser.add_argument('--optimize', choices=['0', '1', '2', '3', 's'], default=None,
                        help='Override optimization level: 0/1/2/3/s')
    parser.add_argument('-e', '--edit', help='Edit config: KEY=VALUE (e.g., ARMCC_PATH=D:/path)')
    parser.add_argument('-sc', '--show-config', action='store_true', help='Show toolchain and include paths')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # init language as early as possible
    set_language(args.lang or get_language())

    if args.edit:
        return 0 if edit_config(args.edit) else 1

    if args.show_config:
        config = load_config()
        print(t('cli.show_config.toolchains'))
        for key, value in config['TOOLCHAINS'].items():
            print(f'  {key} = {value}')
        print('\n' + t('cli.show_config.includes'))
        for key, value in config['INCLUDES'].items():
            print(f'  {key} = {value}')
        if 'NINJA' in config:
            print('\n' + t('cli.show_config.ninja'))
            for key, value in config['NINJA'].items():
                print(f'  {key} = {value}')
        print('\n' + t('cli.show_config.cmake'))
        for key, value in config['CMAKE'].items():
            print(f'  {key} = {value}')
        return 0

    if args.clean and (not args.uvprojx) and (not args.output):
        parser.error(t('cli.error.clean_requires_target'))
        return 1

    if not args.uvprojx and not args.clean:
        parser.error(t('cli.error.uvprojx_required'))
        return 1

    project_data = None
    if args.uvprojx:
        if not os.path.exists(args.uvprojx):
            print(t('cli.error.file_not_found', path=args.uvprojx))
            return 1
        project_data = parse_uvprojx(args.uvprojx)

        if args.compiler:
            project_data['use_armclang'] = (args.compiler == 'armclang')
            project_data['force_compiler'] = args.compiler
        else:
            project_data['force_compiler'] = None

        if args.optimize:
            project_data['opt_level'] = args.optimize

    if args.uvprojx:
        project_root = compute_project_root(args.output, args.uvprojx)
    else:
        project_root = os.path.abspath(args.output)
    os.makedirs(project_root, exist_ok=True)

    if args.clean:
        clean_generated(project_root)
    if args.clean and project_data is None:
        return 0

    generate_cmake_structure(project_data, project_root)
    generate_toolchains(project_data, project_root)

    # Linker templates are generated into the build directory by the toolchain when needed.

    default_compiler = project_data.get('force_compiler') or ('armclang' if project_data['use_armclang'] else 'armcc')
    generate_cmake_presets(project_root, default_compiler)

    clangd_compiler = project_data.get('force_compiler') or ('armclang' if project_data['use_armclang'] else 'armcc')
    generate_clangd_config(project_root, clangd_compiler, detect_cpu_architecture(project_data['device']))

    print('\n' + t('cli.done'))
    print(f"  {t('cli.summary.project')}: {project_data['project_name']}")
    print(f"  {t('cli.summary.device')}: {project_data['device']}")
    print(f"  {t('cli.summary.compiler')}: {clangd_compiler}")
    print(f"  {t('cli.summary.optimize')}: -O{project_data['opt_level']}")
    print(f"  {t('cli.summary.output')}: {os.path.abspath(project_root)}")

    print('\n' + t('cli.build_cmds'))
    print('  cmake --preset keil2cmake')
    print('  cmake --build --preset keil2cmake')
    print('  # or explicitly:')
    print('  cmake --preset keil2cmake-armcc')
    print('  cmake --preset keil2cmake-armclang')
    print('  cmake --preset keil2cmake-armgcc')
    return 0
