# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'

import sys
sys.path.insert(0, str(SRC))

from keil2cmake.common import expand_path, norm_path, ensure_dir, remove_bom_from_file
from keil2cmake.i18n import normalize_lang, set_language, t
from keil2cmake.keil import config as kcfg
from keil2cmake.keil.device import (
    detect_cpu_architecture,
    get_compiler_cpu_name,
    get_arm_arch_for_clang,
)
from keil2cmake.compiler.armgcc.layout import (
    infer_sysroot_from_armgcc_path,
    infer_gcc_internal_includes_from_armgcc_path,
)
from keil2cmake.template_engine import render_template, write_template
from keil2cmake.compiler.clangd import generate_clangd_config, _infer_gcc_toolchain_root
from keil2cmake.compiler.presets import generate_cmake_presets
from keil2cmake.compiler.debug import infer_openocd_target, generate_debug_templates
from keil2cmake.compiler.toolchains import generate_toolchains
from keil2cmake.project_gen import (
    _relativize_paths,
    _is_gas_source,
    _device_token,
    _find_gcc_startup_candidate,
    _resolve_sources,
    generate_cmake_structure,
    clean_generated,
)
from keil2cmake.keil import scatter as sc
from keil2cmake.keil.uvprojx import parse_uvprojx
from keil2cmake.cli import main as cli_main


class TestCommonUtilities(unittest.TestCase):
    def test_expand_path_env_and_user(self) -> None:
        os.environ['K2C_TEST_ENV'] = 'C:/Temp'
        self.assertEqual(expand_path('%K2C_TEST_ENV%/bin'), 'C:/Temp/bin')
        self.assertTrue(expand_path('~').startswith(os.path.expanduser('~')))

    def test_expand_path_empty(self) -> None:
        self.assertEqual(expand_path(''), '')
        self.assertEqual(expand_path(None), '')

    def test_norm_path(self) -> None:
        self.assertEqual(norm_path('a\\b\\c'), 'a/b/c')

    def test_ensure_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, 'a', 'b', 'c')
            ensure_dir(target)
            self.assertTrue(os.path.isdir(target))

    def test_remove_bom_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bom_path = os.path.join(td, 'bom.txt')
            with open(bom_path, 'wb') as f:
                f.write(b'\xef\xbb\xbfhello')
            self.assertTrue(remove_bom_from_file(bom_path))
            with open(bom_path, 'rb') as f:
                self.assertEqual(f.read(), b'hello')

            no_bom_path = os.path.join(td, 'plain.txt')
            with open(no_bom_path, 'w', encoding='utf-8') as f:
                f.write('plain')
            self.assertFalse(remove_bom_from_file(no_bom_path))

            self.assertFalse(remove_bom_from_file(os.path.join(td, 'missing.txt')))


class TestI18nHelpers(unittest.TestCase):
    def test_normalize_lang(self) -> None:
        self.assertEqual(normalize_lang('zh-cn'), 'zh')
        self.assertEqual(normalize_lang('EN-US'), 'en')
        self.assertEqual(normalize_lang('unknown'), 'zh')

    def test_t_format_fallback(self) -> None:
        set_language('zh')
        self.assertEqual(t('missing.key'), 'missing.key')
        self.assertEqual(t('config.error.format', value='X'), '错误: 编辑格式应为 KEY=VALUE，实际为 X')


class TestConfigPathsAndEdit(unittest.TestCase):
    def test_get_config_path_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, 'path.cfg')
            os.environ['KEIL2CMAKE_CONFIG_PATH'] = cfg
            self.assertEqual(kcfg.get_config_path(), cfg)

    def test_edit_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, 'path.cfg')
            os.environ['KEIL2CMAKE_CONFIG_PATH'] = cfg

            self.assertFalse(kcfg.edit_config('INVALID'))
            self.assertFalse(kcfg.edit_config('BADKEY=1'))
            self.assertTrue(kcfg.edit_config('ARMGCC_PATH=C:/Toolchains/armgcc/bin'))

            loaded = kcfg.load_config()
            self.assertIn('PATHS', loaded)
            self.assertEqual(
                loaded['PATHS'].get('ARMGCC_PATH', ''),
                'C:/Toolchains/armgcc/bin',
            )

    def test_get_openocd_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, 'path.cfg')
            os.environ['KEIL2CMAKE_CONFIG_PATH'] = cfg
            kcfg.edit_config('OPENOCD_PATH=C:/Tools/openocd/bin/openocd.exe')
            self.assertEqual(
                kcfg.get_openocd_path(),
                'C:/Tools/openocd/bin/openocd.exe',
            )


class TestDeviceMapping(unittest.TestCase):
    def test_detect_cpu_architecture(self) -> None:
        self.assertEqual(detect_cpu_architecture('STM32F103C8'), 'Cortex-M3')
        self.assertEqual(detect_cpu_architecture('STM32H743'), 'Cortex-M7')
        self.assertEqual(detect_cpu_architecture('UNKNOWN'), 'Cortex-M4')

    def test_compiler_cpu_name(self) -> None:
        self.assertEqual(get_compiler_cpu_name('Cortex-M4'), 'cortex-m4')
        self.assertEqual(get_compiler_cpu_name('Unknown'), 'cortex-m4')

    def test_arm_arch_for_clang(self) -> None:
        self.assertEqual(get_arm_arch_for_clang('Cortex-M0'), 'armv6-m')
        self.assertEqual(get_arm_arch_for_clang('Unknown'), 'armv7e-m')


class TestArmgccLayout(unittest.TestCase):
    def test_infer_sysroot_from_armgcc_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'toolchain'
            sysroot = root / 'arm-none-eabi'
            include_dir = sysroot / 'include'
            include_dir.mkdir(parents=True, exist_ok=True)
            bin_dir = sysroot / 'bin'
            bin_dir.mkdir(parents=True, exist_ok=True)
            gcc_path = bin_dir / 'arm-none-eabi-gcc.exe'
            gcc_path.write_text('stub', encoding='utf-8')
            inferred = infer_sysroot_from_armgcc_path(str(gcc_path))
            self.assertEqual(Path(inferred).name, 'arm-none-eabi')

    def test_infer_gcc_internal_includes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'toolchain'
            bin_dir = root / 'bin'
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / 'arm-none-eabi-gcc.exe').write_text('stub', encoding='utf-8')
            include_dir = root / 'lib' / 'gcc' / 'arm-none-eabi' / '12.2.0' / 'include'
            include_fixed = root / 'lib' / 'gcc' / 'arm-none-eabi' / '12.2.0' / 'include-fixed'
            include_dir.mkdir(parents=True, exist_ok=True)
            include_fixed.mkdir(parents=True, exist_ok=True)
            found = infer_gcc_internal_includes_from_armgcc_path(str(bin_dir))
            self.assertTrue(any('include' in p for p in found))
            self.assertTrue(any('include-fixed' in p for p in found))


class TestTemplateEngine(unittest.TestCase):
    def test_render_template(self) -> None:
        content = render_template('openocd.cfg.in.j2', {})
        self.assertIn('OpenOCD config', content)

    def test_write_template(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'openocd.cfg')
            write_template('openocd.cfg.in.j2', {}, out, encoding='utf-8')
            self.assertTrue(os.path.exists(out))


class TestClangdConfig(unittest.TestCase):
    def test_generate_clangd_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, 'path.cfg')
            os.environ['KEIL2CMAKE_CONFIG_PATH'] = cfg

            toolchain = Path(td) / 'toolchain'
            sysroot = toolchain / 'arm-none-eabi'
            (sysroot / 'include').mkdir(parents=True, exist_ok=True)
            bin_dir = toolchain / 'bin'
            bin_dir.mkdir(parents=True, exist_ok=True)
            kcfg.edit_config(f'ARMGCC_PATH={bin_dir}')

            project_root = Path(td) / 'proj'
            project_root.mkdir(parents=True, exist_ok=True)
            generate_clangd_config(str(project_root), 'Cortex-M4', use_microlib=True)
            clangd_path = project_root / '.clangd'
            self.assertTrue(clangd_path.exists())
            content = clangd_path.read_text(encoding='utf-8')
            self.assertIn('--target=arm-none-eabi', content)
            self.assertIn('-D__MICROLIB', content)

    def test_infer_gcc_toolchain_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'tc'
            gcc_dir = root / 'lib' / 'gcc' / 'arm-none-eabi'
            gcc_dir.mkdir(parents=True, exist_ok=True)
            bin_dir = root / 'bin'
            bin_dir.mkdir(parents=True, exist_ok=True)
            exe = bin_dir / 'arm-none-eabi-gcc.exe'
            exe.write_text('stub', encoding='utf-8')
            inferred = _infer_gcc_toolchain_root(str(exe))
            self.assertTrue(inferred.endswith('/tc') or inferred.endswith('\\tc'))


class TestProjectGenInternals(unittest.TestCase):
    def test_relativize_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'proj'
            root.mkdir(parents=True, exist_ok=True)
            uv_dir = root / 'MDK-ARM'
            uv_dir.mkdir(parents=True, exist_ok=True)
            abs_file = root / 'src' / 'main.c'
            abs_file.parent.mkdir(parents=True, exist_ok=True)
            abs_file.write_text('int main(){}', encoding='utf-8')
            out = _relativize_paths([str(abs_file), 'rel.c'], str(root), str(uv_dir))
            self.assertIn('src/main.c', out[0])
            self.assertTrue(out[1].endswith('rel.c'))

    def test_is_gas_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / 'startup.s'
            good.write_text('; comment\n.syntax unified\n', encoding='utf-8')
            self.assertTrue(_is_gas_source(str(good)))
            bad = Path(td) / 'bad.s'
            bad.write_text('MOV R0, R1\n', encoding='utf-8')
            self.assertFalse(_is_gas_source(str(bad)))
            self.assertFalse(_is_gas_source(str(Path(td) / 'missing.s')))

    def test_device_token(self) -> None:
        self.assertEqual(_device_token('STM32F103C8'), 'stm32f103c8')
        self.assertEqual(_device_token(''), '')

    def test_find_gcc_startup_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            uv = Path(td) / 'uv'
            uv.mkdir(parents=True, exist_ok=True)
            f1 = uv / 'startup_stm32f103x.s'
            f1.write_text('.syntax unified\n.section .isr_vector\n.global Reset_Handler\n', encoding='utf-8')
            f2 = uv / 'startup_other.s'
            f2.write_text('.syntax unified\n', encoding='utf-8')
            best = _find_gcc_startup_candidate(str(uv), 'STM32F103C8')
            self.assertTrue(best.endswith('startup_stm32f103x.s'))

    def test_resolve_sources_with_asm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'proj'
            uv = root / 'MDK-ARM'
            uv.mkdir(parents=True, exist_ok=True)
            sfile = uv / 'main.s'
            sfile.write_text('.syntax unified\n', encoding='utf-8')
            cfile = uv / 'main.c'
            cfile.write_text('int main(){}', encoding='utf-8')
            data = {
                'uvprojx_dir': str(uv),
                'source_files': ['main.s', 'main.c'],
                'device': 'STM32F103C8',
            }
            sources, asm_sources, gcc_startup = _resolve_sources(data, str(root))
            self.assertIn('main.c', sources[1])
            self.assertTrue(any(x.endswith('main.s') for x in asm_sources))
            self.assertTrue(gcc_startup == '' or gcc_startup.endswith('.s'))

    def test_generate_cmake_structure_and_clean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'proj'
            uv = root / 'MDK-ARM'
            uv.mkdir(parents=True, exist_ok=True)
            data = {
                'project_name': 'demo',
                'device': 'STM32F103C8',
                'uvprojx_dir': str(uv),
                'source_files': [],
                'include_paths': [],
                'defines': [],
                'c_flags': '',
                'asm_flags': '',
                'ld_flags': '',
                'keil_optim': '11',
                'keil_compiler': 'armcc',
                'use_microlib': False,
            }
            root.mkdir(parents=True, exist_ok=True)
            generate_cmake_structure(data, str(root))
            self.assertTrue((root / 'CMakeLists.txt').exists())

            # create additional files to be cleaned
            (root / 'CMakePresets.json').write_text('{}', encoding='utf-8')
            (root / '.clangd').write_text('CompileFlags', encoding='utf-8')
            vscode = root / '.vscode'
            vscode.mkdir(parents=True, exist_ok=True)
            (vscode / 'launch.json').write_text('{}', encoding='utf-8')
            (vscode / 'tasks.json').write_text('{}', encoding='utf-8')
            internal = root / 'cmake' / 'internal' / 'templates'
            internal.mkdir(parents=True, exist_ok=True)
            (internal / 'openocd.cfg.in').write_text('#', encoding='utf-8')
            (root / 'cmake' / 'user' / 'openocd.cfg').write_text('#', encoding='utf-8')

            clean_generated(str(root))
            self.assertFalse((root / 'CMakeLists.txt').exists())
            self.assertFalse((root / 'CMakePresets.json').exists())
            self.assertFalse((vscode / 'launch.json').exists())

            # clean empty project (no files)
            clean_generated(str(root))


class TestScatterInternals(unittest.TestCase):
    def test_safe_eval_expr_variants(self) -> None:
        self.assertIsNone(sc._safe_eval_expr('', {}))
        self.assertIsNone(sc._safe_eval_expr('+1', {}))
        self.assertEqual(sc._safe_eval_expr('1+2', {}), 3)
        self.assertIsNone(sc._safe_eval_expr('UNKNOWN', {}))
        self.assertIsNone(sc._safe_eval_expr('A', {'A': 'A'}))
        self.assertIsNone(sc._safe_eval_expr('1**2', {}))

    def test_normalize_and_format(self) -> None:
        self.assertEqual(sc._normalize_expr(' 1K '), '(1 * 1024)')
        self.assertEqual(sc._format_hex(None), '0x0')

    def test_convert_missing_file(self) -> None:
        result = sc.convert_scatter_to_ld('missing.sct', 'out.ld')
        self.assertFalse(result.ok)


class TestPresetsAndDebugTemplates(unittest.TestCase):
    def test_generate_presets_and_debug_templates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, 'path.cfg')
            os.environ['KEIL2CMAKE_CONFIG_PATH'] = cfg
            project_root = Path(td) / 'proj'
            project_root.mkdir(parents=True, exist_ok=True)

            generate_cmake_presets(str(project_root), {'device': 'STM32F103C8'})
            presets = (project_root / 'CMakePresets.json').read_text(encoding='utf-8')
            self.assertIn('K2C_OPENOCD_PATH', presets)
            self.assertIn('target/stm32f1x.cfg', presets)

            generate_debug_templates(str(project_root))
            debug_cmake = project_root / 'cmake' / 'internal' / 'k2c_debug.cmake'
            self.assertTrue(debug_cmake.exists())


class TestUvprojxAndCli(unittest.TestCase):
    def _write_uvprojx(self, path: str, use_ac6: str) -> None:
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Project>
  <Targets>
    <Target>
      <TargetName>demo</TargetName>
      <TargetOption>
        <TargetCommonOption>
          <Device>STM32F103C8</Device>
        </TargetCommonOption>
        <uAC6>{use_ac6}</uAC6>
        <TargetArmAds>
          <Cads>
            <VariousControls>
              <IncludePath>..\\Inc</IncludePath>
              <Define>USE_HAL,TEST=1</Define>
              <MiscControls>-Wall</MiscControls>
            </VariousControls>
            <Optim>2</Optim>
          </Cads>
          <Aads>
            <VariousControls>
              <MiscControls>-g</MiscControls>
            </VariousControls>
          </Aads>
          <LDads>
            <ScatterFile>Template.sct</ScatterFile>
            <VariousControls>
              <MiscControls>-Wl,--gc-sections</MiscControls>
            </VariousControls>
          </LDads>
        </TargetArmAds>
      </TargetOption>
      <Groups>
        <Group>
          <GroupName>Src</GroupName>
          <Files>
            <File>
              <FilePath>main.c</FilePath>
            </File>
          </Files>
        </Group>
      </Groups>
    </Target>
  </Targets>
</Project>
'''
        Path(path).write_text(xml, encoding='utf-8')

    def test_parse_uvprojx_armcc(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            uvprojx = os.path.join(td, 'demo.uvprojx')
            self._write_uvprojx(uvprojx, '0')
            data = parse_uvprojx(uvprojx)
            self.assertEqual(data.get('keil_compiler'), 'armcc')
            self.assertIn('main.c', data.get('source_files', []))
            self.assertIn('USE_HAL', data.get('defines', []))

    def test_cli_main_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = os.path.join(td, 'path.cfg')
            os.environ['KEIL2CMAKE_CONFIG_PATH'] = cfg

            # failure
            self.assertEqual(cli_main(['missing.uvprojx', '-o', os.path.join(td, 'out')]), 1)

            # success
            uvprojx = os.path.join(td, 'demo.uvprojx')
            self._write_uvprojx(uvprojx, '1')
            out_dir = os.path.join(td, 'out')
            ret = cli_main([uvprojx, '-o', out_dir])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.exists(os.path.join(out_dir, 'CMakeLists.txt')))
            self.assertTrue(os.path.exists(os.path.join(out_dir, 'cmake', 'internal', 'k2c_debug.cmake')))


class TestToolchains(unittest.TestCase):
    def test_generate_toolchains_with_sct(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / 'proj'
            uv_dir = Path(td) / 'uv'
            project_root.mkdir(parents=True, exist_ok=True)
            uv_dir.mkdir(parents=True, exist_ok=True)

            sct_path = uv_dir / 'Template.sct'
            sct_path.write_text('#define __ROM_BASE 0x08000000\n#define __ROM_SIZE 0x20000\n#define __RAM_BASE 0x20000000\n#define __RAM_SIZE 0x5000\nLR_IROM1 __ROM_BASE __ROM_SIZE { }', encoding='utf-8')

            data = {
                'device': 'STM32F103C8',
                'uvprojx_dir': str(uv_dir),
                'linker_script': 'Template.sct',
            }
            generate_toolchains(data, str(project_root))
            tc = project_root / 'cmake' / 'internal' / 'toolchain.cmake'
            self.assertTrue(tc.exists())
            self.assertIn('keil2cmake_from_sct.ld', tc.read_text(encoding='utf-8'))
