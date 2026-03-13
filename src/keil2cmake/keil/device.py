# -*- coding: utf-8 -*-


def get_compiler_cpu_name(cpu_arch: str) -> str:
    """CPU 架构到编译器常用 -mcpu 名称。"""
    mapping = {
        'Cortex-M0': 'cortex-m0',
        'Cortex-M0+': 'cortex-m0plus',
        'Cortex-M1': 'cortex-m1',
        'Cortex-M3': 'cortex-m3',
        'Cortex-M4': 'cortex-m4',
        'Cortex-M7': 'cortex-m7',
        'Cortex-M23': 'cortex-m23',
        'Cortex-M33': 'cortex-m33',
        'Cortex-M35P': 'cortex-m35p',
        'Cortex-M55': 'cortex-m55',
        'Cortex-M85': 'cortex-m85',
    }
    return mapping.get(cpu_arch, 'cortex-m4')


def detect_cpu_architecture(device_name: str) -> str:
    """根据 STM32 系列名称推断 CPU 架构。"""
    device_upper = (device_name or '').upper()

    series_to_cpu = {
        'STM32F0': 'Cortex-M0',
        'STM32L0': 'Cortex-M0+',
        'STM32G0': 'Cortex-M0+',
        'STM32F1': 'Cortex-M3',
        'STM32L1': 'Cortex-M3',
        'STM32F2': 'Cortex-M3',
        'STM32F3': 'Cortex-M4',
        'STM32L4': 'Cortex-M4',
        'STM32G4': 'Cortex-M4',
        'STM32F4': 'Cortex-M4',
        'STM32L5': 'Cortex-M33',
        'STM32U5': 'Cortex-M33',
        'STM32F7': 'Cortex-M7',
        'STM32H7': 'Cortex-M7',
        'STM32WB': 'Cortex-M4',
        'STM32WL': 'Cortex-M4',
    }

    for series, cpu in series_to_cpu.items():
        if series in device_upper:
            return cpu

    return 'Cortex-M4'


def get_arm_arch_for_clang(cpu_arch: str) -> str:
    """将 CPU 架构转换为 ARMClang 所需的架构版本。"""
    cpu_to_arm_arch = {
        'Cortex-M0': 'armv6-m',
        'Cortex-M0+': 'armv6-m',
        'Cortex-M1': 'armv6-m',
        'Cortex-M3': 'armv7-m',
        'Cortex-M4': 'armv7e-m',
        'Cortex-M7': 'armv7e-m',
        'Cortex-M23': 'armv8-m.base',
        'Cortex-M33': 'armv8-m.main',
        'Cortex-M35P': 'armv8-m.main',
        'Cortex-M55': 'armv8.1-m.main',
        'Cortex-M85': 'armv8.1-m.main',
    }
    return cpu_to_arm_arch.get(cpu_arch, 'armv7e-m')
