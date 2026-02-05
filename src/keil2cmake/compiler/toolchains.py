# -*- coding: utf-8 -*-

import os

from ..keil.config import get_toolchain_path, get_sysroot_path, get_cmake_min_version
from ..keil.device import detect_cpu_architecture, get_compiler_cpu_name
from ..keil.scatter import convert_scatter_to_ld
from ..common import ensure_dir, norm_path, remove_bom_from_file
from ..i18n import t
from ..template_engine import write_template


def generate_toolchains(project_data: dict, project_root: str) -> None:
    """Generate a single toolchain file under cmake/internal/toolchain.cmake (ARM-GCC only)."""
    cpu_arch = detect_cpu_architecture(project_data['device'])
    cpu_name = get_compiler_cpu_name(cpu_arch)

    armgcc_path = get_toolchain_path('ARMGCC')
    armgcc_sysroot = get_sysroot_path()

    internal_dir = os.path.join(project_root, 'cmake', 'internal')
    ensure_dir(internal_dir)

    user_rel = "${CMAKE_CURRENT_LIST_DIR}/../user/keil2cmake_user.cmake"

    ld_template = r'''
/* Auto-generated template for arm-none-eabi-gcc.
 * NOTE: You MUST adjust MEMORY sizes to your MCU.
 */

ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS
{
    .isr_vector :
    {
        KEEP(*(.isr_vector))
    } > FLASH

    .text :
    {
        *(.text*)
        *(.rodata*)
        KEEP(*(.init))
        KEEP(*(.fini))
    } > FLASH

    .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
    .ARM.exidx :
    {
        __exidx_start = .;
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
        __exidx_end = .;
    } > FLASH

    .data : AT (ADDR(.text) + SIZEOF(.text))
    {
        __data_start__ = .;
        *(.data*)
        __data_end__ = .;
    } > RAM

    .bss :
    {
        __bss_start__ = .;
        *(.bss*)
        *(COMMON)
        __bss_end__ = .;
    } > RAM
}
'''.strip()

    default_ld_path = os.path.join(internal_dir, 'keil2cmake_default.ld')
    if not os.path.exists(default_ld_path):
        with open(default_ld_path, 'w', encoding='utf-8-sig') as f:
            f.write(ld_template)

    default_ld_file = 'keil2cmake_default.ld'
    linker_script = project_data.get('linker_script')
    if linker_script:
        uvprojx_dir = project_data.get('uvprojx_dir', '')
        sct_path = linker_script if os.path.isabs(linker_script) else os.path.join(uvprojx_dir, linker_script)
        if os.path.isfile(sct_path) and sct_path.lower().endswith('.sct'):
            converted_path = os.path.join(internal_dir, 'keil2cmake_from_sct.ld')
            result = convert_scatter_to_ld(sct_path, converted_path)
            if result.ok:
                default_ld_file = 'keil2cmake_from_sct.ld'
                print(f"  鈿狅笍  {t('gen.toolchain.sct_converted')}: {os.path.basename(sct_path)}")
            else:
                print(f"  鈿狅笍  {t('gen.toolchain.sct_failed')}: {os.path.basename(sct_path)}")

    write_template(
        'toolchain.cmake.j2',
        {
            'header_title': t('gen.toolchain.header.title'),
            'linker_header': t('gen.toolchain.linker_scripts'),
            'cmake_min_version': get_cmake_min_version(),
            'cpu_name': cpu_name,
            'user_rel': user_rel,
            'armgcc_bin': norm_path(armgcc_path),
            'armgcc_sysroot': norm_path(armgcc_sysroot),
            'default_ld_file': default_ld_file,
        },
        os.path.join(internal_dir, 'toolchain.cmake'),
        encoding='utf-8-sig',
    )

    # Check and clean BOM from user's linker script if provided
    if linker_script:
        uvprojx_dir = project_data.get('uvprojx_dir', '')
        script_path = linker_script if os.path.isabs(linker_script) else os.path.join(uvprojx_dir, linker_script)

        if os.path.exists(script_path):
            if remove_bom_from_file(script_path):
                print(f"  鈿狅笍  {t('gen.toolchain.bom_removed')}: {os.path.basename(script_path)}")
