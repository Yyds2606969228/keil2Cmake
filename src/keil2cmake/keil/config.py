# -*- coding: utf-8 -*-

import configparser
import os
from pathlib import Path

from ..compiler.armgcc.layout import infer_sysroot_from_armgcc_path
from ..i18n import t
from ..common import expand_path


_DEFAULTS_PATHS = {
    'ARMGCC_PATH': '',
    'CMAKE_PATH': 'cmake',
    'NINJA_PATH': 'ninja',
    'CHECKCPP_PATH': 'checkcpp',
    'OPENOCD_PATH': 'openocd',
}

_DEFAULTS_GENERAL = {
    'LANGUAGE': 'zh',
}


def get_config_path() -> str:
    """配置文件路径（用户目录下）。"""
    override = os.environ.get('KEIL2CMAKE_CONFIG_PATH')
    if override:
        return str(Path(override))
    return str(Path.home() / '.config' / 'keil2cmake' / 'path.cfg')


def _apply_defaults(config: configparser.ConfigParser) -> None:
    if 'PATHS' not in config:
        config['PATHS'] = {}
    if 'GENERAL' not in config:
        config['GENERAL'] = {}

    for k, v in _DEFAULTS_PATHS.items():
        config['PATHS'].setdefault(k, v)
    for k, v in _DEFAULTS_GENERAL.items():
        config['GENERAL'].setdefault(k, v)


def load_config() -> configparser.ConfigParser:
    """加载配置文件（纯读取；不存在时仅返回内存默认值）。"""
    config = configparser.ConfigParser()
    cfg_path = get_config_path()

    if os.path.exists(cfg_path):
        config.read(cfg_path, encoding='utf-8')

    _apply_defaults(config)
    return config


def save_config(config: configparser.ConfigParser) -> None:
    _apply_defaults(config)
    cfg_path = get_config_path()
    cfg_dir = os.path.dirname(cfg_path)
    if cfg_dir:
        os.makedirs(cfg_dir, exist_ok=True)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        config.write(f)


def edit_config(edit_string: str) -> bool:
    config = load_config()

    if '=' not in edit_string:
        print(t('config.error.format', value=edit_string))
        return False

    configkey, value = edit_string.split('=', 1)
    configkey = configkey.strip().upper()

    valid_path_keys = ['ARMGCC_PATH', 'CMAKE_PATH', 'NINJA_PATH', 'CHECKCPP_PATH', 'OPENOCD_PATH']
    valid_general_keys = ['LANGUAGE']
    valid_keys = valid_path_keys + valid_general_keys

    if configkey not in valid_keys:
        print(t('config.error.invalid_key', configkey=configkey, valid=', '.join(valid_keys)))
        return False

    if configkey in valid_path_keys:
        config['PATHS'][configkey] = value
    else:
        config['GENERAL'][configkey] = value

    save_config(config)
    print(t('config.updated', configkey=configkey, value=value))
    return True


def get_language() -> str:
    config = load_config()
    return str(config['GENERAL'].get('LANGUAGE', 'zh')).strip() or 'zh'


def get_toolchain_path(toolchain_type: str) -> str:
    if str(toolchain_type).strip().upper() != 'ARMGCC':
        return ''
    config = load_config()
    return expand_path(config['PATHS'].get('ARMGCC_PATH', ''))


def get_armgcc_path() -> str:
    config = load_config()
    return expand_path(config['PATHS'].get('ARMGCC_PATH', ''))


def get_sysroot_path() -> str:
    armgcc_path = get_armgcc_path()
    inferred = infer_sysroot_from_armgcc_path(armgcc_path)
    return expand_path(inferred)


def get_ninja_path() -> str:
    config = load_config()
    return expand_path(config['PATHS'].get('NINJA_PATH', 'ninja'))


def get_cmake_path() -> str:
    config = load_config()
    return expand_path(config['PATHS'].get('CMAKE_PATH', 'cmake'))


def get_checkcpp_path() -> str:
    config = load_config()
    return expand_path(config['PATHS'].get('CHECKCPP_PATH', 'checkcpp'))


def get_openocd_path() -> str:
    config = load_config()
    return expand_path(config['PATHS'].get('OPENOCD_PATH', 'openocd'))


def get_cmake_min_version() -> str:
    return '3.23'
