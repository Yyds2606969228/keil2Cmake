# -*- coding: utf-8 -*-

import configparser
import os
from pathlib import Path

from keil2cmake_common import expand_path, SUPPORTED_COMPILERS
from compiler.armgcc.layout import infer_sysroot_from_armgcc_path
from i18n import t


def get_config_path() -> str:
    """配置文件路径（用户目录下）。"""
    override = os.environ.get('KEIL2CMAKE_CONFIG_PATH')
    if override:
        return str(Path(override))
    return str(Path.home() / '.keil2cmake' / 'path.cfg')


def load_config() -> configparser.ConfigParser:
    """加载配置文件（不存在则创建默认）。"""
    config = configparser.ConfigParser()
    cfg_path = get_config_path()

    # defaults (migrate/ensure keys)
    defaults_toolchains = {
        'ARMCC_PATH': 'D:/Program/Keil_v5/ARM/ARMCC/bin/',
        'ARMCLANG_PATH': 'D:/Program/Keil_v5/ARM/ARMCLANG/bin/',
        'ARMGCC_PATH': 'D:/Program/GNUArmEmbeddedToolchain/bin/',
    }
    defaults_includes = {
        'ARMCC_INCLUDE': 'D:/Program/Keil_v5/ARM/ARMCC/include/',
        'ARMCLANG_INCLUDE': 'D:/Program/Keil_v5/ARM/ARMCLANG/include/',
        'ARMGCC_SYSROOT': '',
        'ARMGCC_INCLUDE': '',
    }
    defaults_ninja = {
        'ENABLED': '1',
        'PATH': 'ninja',
    }
    defaults_cmake = {
        'MIN_VERSION': '3.20',
    }
    defaults_general = {
        'LANGUAGE': 'zh',
    }

    if os.path.exists(cfg_path):
        config.read(cfg_path, encoding='utf-8')

    if 'TOOLCHAINS' not in config:
        config['TOOLCHAINS'] = {}
    if 'INCLUDES' not in config:
        config['INCLUDES'] = {}
    if 'NINJA' not in config:
        config['NINJA'] = {}
    if 'CMAKE' not in config:
        config['CMAKE'] = {}
    if 'GENERAL' not in config:
        config['GENERAL'] = {}

    for k, v in defaults_toolchains.items():
        config['TOOLCHAINS'].setdefault(k, v)
    for k, v in defaults_includes.items():
        config['INCLUDES'].setdefault(k, v)
    for k, v in defaults_ninja.items():
        config['NINJA'].setdefault(k, v)
    for k, v in defaults_cmake.items():
        config['CMAKE'].setdefault(k, v)
    for k, v in defaults_general.items():
        config['GENERAL'].setdefault(k, v)

    # ensure directory exists
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        config.write(f)

    return config


def save_config(config: configparser.ConfigParser) -> None:
    with open(get_config_path(), 'w', encoding='utf-8') as f:
        config.write(f)


def edit_config(edit_string: str) -> bool:
    config = load_config()

    if '=' not in edit_string:
        print(t('config.error.format', value=edit_string))
        return False

    configkey, value = edit_string.split('=', 1)
    configkey = configkey.strip().upper()

    valid_toolchain_keys = ['ARMCC_PATH', 'ARMCLANG_PATH', 'ARMGCC_PATH']
    valid_include_keys = ['ARMCC_INCLUDE', 'ARMCLANG_INCLUDE', 'ARMGCC_SYSROOT', 'ARMGCC_INCLUDE']
    valid_cmake_keys = ['MIN_VERSION']
    valid_ninja_keys = ['NINJA_ENABLED', 'NINJA_PATH']
    valid_general_keys = ['LANGUAGE']

    valid_keys = valid_toolchain_keys + valid_include_keys + valid_cmake_keys + valid_ninja_keys + valid_general_keys

    if configkey not in valid_keys:
        print(t('config.error.invalid_key', configkey=configkey, valid=', '.join(valid_keys)))
        return False

    if configkey in valid_toolchain_keys:
        config['TOOLCHAINS'][configkey] = value
    elif configkey in valid_include_keys:
        config['INCLUDES'][configkey] = value
    elif configkey in valid_cmake_keys:
        config['CMAKE'][configkey] = value
    elif configkey in valid_ninja_keys:
        if configkey == 'NINJA_ENABLED':
            config['NINJA']['ENABLED'] = value
        elif configkey == 'NINJA_PATH':
            config['NINJA']['PATH'] = value
    elif configkey in valid_general_keys:
        config['GENERAL'][configkey] = value

    save_config(config)
    print(t('config.updated', configkey=configkey, value=value))
    return True


def get_language() -> str:
    config = load_config()
    return str(config['GENERAL'].get('LANGUAGE', 'zh')).strip() or 'zh'


def get_toolchain_path(toolchain_type: str) -> str:
    config = load_config()
    key = f"{toolchain_type}_PATH"
    return expand_path(config['TOOLCHAINS'].get(key, ''))


def get_include_path(compiler_type: str) -> str:
    config = load_config()
    key = f"{compiler_type}_INCLUDE"
    return expand_path(config['INCLUDES'].get(key, ''))


def get_sysroot_path() -> str:
    config = load_config()
    configured = expand_path(config['INCLUDES'].get('ARMGCC_SYSROOT', ''))
    if configured:
        return configured

    # fallback: infer from ARMGCC_PATH
    armgcc_path = get_toolchain_path('ARMGCC')
    inferred = infer_sysroot_from_armgcc_path(armgcc_path)
    return expand_path(inferred)


def get_armgcc_extra_include() -> str:
    config = load_config()
    return expand_path(config['INCLUDES'].get('ARMGCC_INCLUDE', ''))


def get_ninja_enabled() -> bool:
    config = load_config()
    enabled = config['NINJA'].get('ENABLED', '1')
    return str(enabled).strip() not in ('0', 'false', 'False', 'no', 'NO')


def get_ninja_path() -> str:
    config = load_config()
    return expand_path(config['NINJA'].get('PATH', 'ninja'))


def get_cmake_min_version() -> str:
    config = load_config()
    return config['CMAKE'].get('MIN_VERSION', '3.20')
