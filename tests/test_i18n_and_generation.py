# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from keil2cmake.i18n import set_language, t


class TestI18n(unittest.TestCase):
    def test_translate_basic(self) -> None:
        set_language('en')
        self.assertEqual(
            t('cli.help.description'),
            'Keil uVision to CMake converter (ARM-GCC only, with clangd support)'
        )
        set_language('zh')
        self.assertEqual(
            t('cli.help.description'),
            'Keil uVision 转 CMake 工具（仅 ARM-GCC，含 clangd 支持）'
        )


class TestConfigPaths(unittest.TestCase):
    def test_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, 'path.cfg')
            os.environ['KEIL2CMAKE_CONFIG_PATH'] = cfg_path

            from keil2cmake.keil import config as kcfg

            cfg = kcfg.load_config()
            self.assertIn('PATHS', cfg)
            self.assertEqual(cfg['PATHS'].get('ARMGCC_PATH', ''), '')
            self.assertEqual(cfg['PATHS'].get('CMAKE_PATH', ''), 'cmake')
            self.assertEqual(cfg['PATHS'].get('NINJA_PATH', ''), 'ninja')
            self.assertEqual(cfg['PATHS'].get('CHECKCPP_PATH', ''), 'checkcpp')
            self.assertEqual(cfg['PATHS'].get('OPENOCD_PATH', ''), 'openocd')
            self.assertFalse(os.path.exists(cfg_path))

            kcfg.save_config(cfg)
            self.assertTrue(os.path.exists(cfg_path))


class TestUvprojxAndGeneration(unittest.TestCase):
    def test_generation_relativized_paths_and_presets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = os.path.join(td, 'demo')
            mdk_dir = os.path.join(project_root, 'MDK-ARM')
            os.makedirs(mdk_dir, exist_ok=True)

            # Create ARMASM startup (Keil) and GCC startup candidate
            arm_startup_rel = os.path.join('..', 'Core', 'startup_armcc.s')
            arm_startup_abs = os.path.join(project_root, 'Core', 'startup_armcc.s')
            os.makedirs(os.path.dirname(arm_startup_abs), exist_ok=True)
            with open(arm_startup_abs, 'w', encoding='utf-8') as f:
                f.write('AREA |.text|, CODE, READONLY\\nEXPORT Reset_Handler\\nEND\\n')

            gcc_startup_abs = os.path.join(project_root, 'Core', 'startup_stm32f103x.s')
            with open(gcc_startup_abs, 'w', encoding='utf-8') as f:
                f.write('.syntax unified\\n.section .isr_vector\\n.global Reset_Handler\\n')

            uvprojx_path = os.path.join(mdk_dir, 'qr.uvprojx')
            sct_path = os.path.join(mdk_dir, 'Template.sct')
            sct_content = '''#define __ROM_BASE  0x08000000
#define __ROM_SIZE  0x00020000
#define __RAM_BASE  0x20000000
#define __RAM_SIZE  0x00005000
#define __STACK_SIZE 0x00000800
#define __HEAP_SIZE  0x00000400

LR_IROM1 __ROM_BASE __ROM_SIZE
{
  ER_IROM1 __ROM_BASE __ROM_SIZE
  {
    *.o (RESET, +First)
    .ANY (+RO)
  }
  RW_IRAM1 __RAM_BASE __RAM_SIZE
  {
    .ANY (+RW +ZI)
  }
}
'''
            with open(sct_path, 'w', encoding='utf-8') as f:
                f.write(sct_content)

            uvprojx_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Project>
  <Targets>
    <Target>
      <TargetName>qr</TargetName>
      <TargetOption>
        <TargetCommonOption>
          <Device>STM32F103C8</Device>
          <OutputDirectory>build/</OutputDirectory>
          <UseMicroLIB>1</UseMicroLIB>
        </TargetCommonOption>
        <uAC6>1</uAC6>
        <TargetArmAds>
          <Cads>
            <Optimization>0</Optimization>
            <VariousControls>
              <IncludePath>..\\Core</IncludePath>
              <Define>USE_HAL,TEST=1</Define>
              <MiscControls>-Wall</MiscControls>
            </VariousControls>
          </Cads>
          <Aads>
            <VariousControls>
              <MiscControls></MiscControls>
            </VariousControls>
          </Aads>
          <LDads>
            <ScatterFile>..\\MDK-ARM\\Template.sct</ScatterFile>
            <VariousControls>
              <MiscControls></MiscControls>
            </VariousControls>
          </LDads>
        </TargetArmAds>
        <DebugOpt>
          <pMon>Segger\\JL2CM3.dll</pMon>
        </DebugOpt>
      </TargetOption>
      <Groups>
        <Group>
          <GroupName>Startup</GroupName>
          <Files>
            <File>
              <FilePath>{arm_startup_rel}</FilePath>
            </File>
          </Files>
        </Group>
      </Groups>
    </Target>
  </Targets>
</Project>
'''
            with open(uvprojx_path, 'w', encoding='utf-8') as f:
                f.write(uvprojx_xml)

            os.environ['KEIL2CMAKE_CONFIG_PATH'] = os.path.join(td, 'path.cfg')

            from keil2cmake.keil.uvprojx import parse_uvprojx
            from keil2cmake.project_gen import generate_cmake_structure
            from keil2cmake.compiler.toolchains import generate_toolchains
            from keil2cmake.compiler.presets import generate_cmake_presets
            from keil2cmake.compiler.debug import generate_openocd_files

            set_language('en')
            data = parse_uvprojx(uvprojx_path)
            self.assertTrue(data.get('uvprojx_dir', '').endswith('MDK-ARM'))
            self.assertEqual(data.get('keil_compiler'), 'armclang')
            self.assertEqual(data.get('debugger'), 'jlink')
            self.assertTrue(data.get('use_microlib'))

            os.makedirs(project_root, exist_ok=True)
            generate_cmake_structure(data, project_root)
            generate_toolchains(data, project_root)
            generate_cmake_presets(project_root, data)
            generate_openocd_files(project_root, data.get('device', ''), data.get('debugger', ''))

            gen_path = os.path.join(project_root, 'cmake', 'user', 'keil2cmake_user.cmake')
            self.assertTrue(os.path.exists(gen_path))
            with open(gen_path, 'r', encoding='utf-8') as f:
                gen = f.read()

            # The ASM source should be included; GCC startup should be suggested but not included by default.
            self.assertIn('startup_armcc.s', gen)
            self.assertIn('startup_stm32f103x.s', gen)
            self.assertIn('set(K2C_USE_NEWLIB_NANO ON', gen)
            self.assertIn('set(K2C_ASM_DETECTED ON', gen)
            self.assertNotIn('set(K2C_CHECKCPP_ENABLE', gen)
            self.assertNotIn('set(K2C_OPENOCD_PATH', gen)
            self.assertNotIn('set(K2C_DEBUG_PROBE', gen)

            cppcheck_path = os.path.join(project_root, 'cmake', 'user', 'cppcheck.cmake')
            self.assertTrue(os.path.exists(cppcheck_path))
            with open(cppcheck_path, 'r', encoding='utf-8') as f:
                cppcheck_cfg = f.read()
            self.assertIn('set(K2C_CHECKCPP_ENABLE', cppcheck_cfg)
            self.assertIn('set(K2C_CHECKCPP_JOBS', cppcheck_cfg)
            self.assertIn('set(K2C_CHECKCPP_EXCLUDES', cppcheck_cfg)
            self.assertIn('set(K2C_CHECKCPP_ENABLE_ALL', cppcheck_cfg)
            self.assertIn('set(K2C_CHECKCPP_ENABLE_WARNING', cppcheck_cfg)
            self.assertIn('set(K2C_CHECKCPP_INCONCLUSIVE', cppcheck_cfg)

            presets_path = os.path.join(project_root, 'CMakePresets.json')
            with open(presets_path, 'r', encoding='utf-8') as f:
                presets = f.read()
            self.assertIn('"name": "build"', presets)
            self.assertIn('"name": "check"', presets)
            self.assertIn('K2C_CHECKCPP', presets)
            self.assertIn('K2C_OPENOCD_PATH', presets)
            self.assertIn('K2C_OPENOCD_TARGET', presets)
            self.assertIn('K2C_OPENOCD_INTERFACE', presets)
            self.assertIn('K2C_OPENOCD_TRANSPORT', presets)
            self.assertIn('target/stm32f1x.cfg', presets)
            self.assertIn('interface/jlink.cfg', presets)
            self.assertIn('"K2C_OPENOCD_TRANSPORT": "swd"', presets)
            self.assertNotIn('K2C_DEBUG_PROBE', presets)

            # Ensure we keep ASM optimization isolation in top-level CMakeLists.
            cmakelists = os.path.join(project_root, 'CMakeLists.txt')
            with open(cmakelists, 'r', encoding='utf-8') as f:
                top = f.read()
            self.assertIn('$<$<COMPILE_LANGUAGE:C>:', top)
            self.assertIn('$<$<COMPILE_LANGUAGE:CXX>:', top)
            self.assertIn('K2C_CHECKCPP_ENABLE', top)
            self.assertIn('K2C_CHECKCPP_JOBS', top)
            self.assertIn('K2C_CHECKCPP_EXCLUDES', top)
            self.assertIn('cmake/user/cppcheck.cmake', top)
            self.assertNotIn('k2c_debug.cmake', top)

            launch_path = os.path.join(project_root, '.vscode', 'launch.json')
            tasks_path = os.path.join(project_root, '.vscode', 'tasks.json')
            ocd_cfg = os.path.join(project_root, 'cmake', 'user', 'openocd.cfg')
            self.assertTrue(os.path.exists(launch_path))
            self.assertTrue(os.path.exists(tasks_path))
            self.assertTrue(os.path.exists(ocd_cfg))
            with open(ocd_cfg, 'r', encoding='utf-8') as f:
                ocd = f.read()
            self.assertIn('interface/jlink.cfg', ocd)
            self.assertIn('transport select swd', ocd)

            # Scatter conversion should generate ld script and set default linker script.
            converted_ld = os.path.join(project_root, 'cmake', 'internal', 'keil2cmake_from_sct.ld')
            self.assertTrue(os.path.exists(converted_ld))
            with open(converted_ld, 'r', encoding='utf-8') as f:
                ld = f.read()
            self.assertIn('ORIGIN = 0x08000000', ld)
            self.assertIn('LENGTH = 0x00020000', ld)
            self.assertIn('ORIGIN = 0x20000000', ld)
            self.assertIn('LENGTH = 0x00005000', ld)

            toolchain = os.path.join(project_root, 'cmake', 'internal', 'toolchain.cmake')
            with open(toolchain, 'r', encoding='utf-8') as f:
                tc = f.read()
            self.assertIn('keil2cmake_from_sct.ld', tc)
            self.assertIn('set(CMAKE_EXECUTABLE_SUFFIX ".elf")', tc)


class TestScatterConversion(unittest.TestCase):
    def test_scatter_convert_with_macros_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sct_path = os.path.join(td, 'Template.sct')
            out_path = os.path.join(td, 'from_sct.ld')
            sct_content = '''#define ROM_BASE 0x08000000
#define ROM_SIZE 128K
#define RAM_BASE 0x20000000
#define RAM_SIZE (20K + 4K)
#define __STACK_SIZE 0x1000
#define __HEAP_SIZE  0x800

LR_IROM1 ROM_BASE ROM_SIZE
{
  ER_IROM1 ROM_BASE ROM_SIZE
  {
    .ANY (+RO)
  }
  RW_IRAM1 RAM_BASE RAM_SIZE
  {
    .ANY (+RW +ZI)
  }
}
'''
            with open(sct_path, 'w', encoding='utf-8') as f:
                f.write(sct_content)

            from keil2cmake.keil.scatter import convert_scatter_to_ld

            result = convert_scatter_to_ld(sct_path, out_path)
            self.assertTrue(result.ok)
            with open(out_path, 'r', encoding='utf-8') as f:
                ld = f.read()
            self.assertIn('ORIGIN = 0x08000000', ld)
            self.assertIn('LENGTH = 0x00020000', ld)
            self.assertIn('ORIGIN = 0x20000000', ld)
            self.assertIn('LENGTH = 0x00006000', ld)
            self.assertIn('. = . + 0x00000800;', ld)
            self.assertIn('. = . + 0x00001000;', ld)

    def test_scatter_convert_with_regions_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sct_path = os.path.join(td, 'Template.sct')
            out_path = os.path.join(td, 'from_sct.ld')
            sct_content = '''LR_IROM1 0x08000000 0x00040000
{
  ER_IROM1 0x08000000 0x00040000
  {
    .ANY (+RO)
  }
  RW_IRAM1 0x20000000 0x00008000
  {
    .ANY (+RW +ZI)
  }
}
'''
            with open(sct_path, 'w', encoding='utf-8') as f:
                f.write(sct_content)

            from keil2cmake.keil.scatter import convert_scatter_to_ld

            result = convert_scatter_to_ld(sct_path, out_path)
            self.assertTrue(result.ok)
            with open(out_path, 'r', encoding='utf-8') as f:
                ld = f.read()
            self.assertIn('ORIGIN = 0x08000000', ld)
            self.assertIn('LENGTH = 0x00040000', ld)
            self.assertIn('ORIGIN = 0x20000000', ld)
            self.assertIn('LENGTH = 0x00008000', ld)

    def test_scatter_convert_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sct_path = os.path.join(td, 'Template.sct')
            out_path = os.path.join(td, 'from_sct.ld')
            with open(sct_path, 'w', encoding='utf-8') as f:
                f.write('INVALID CONTENT')

            from keil2cmake.keil.scatter import convert_scatter_to_ld

            result = convert_scatter_to_ld(sct_path, out_path)
            self.assertFalse(result.ok)


class TestDebugTemplates(unittest.TestCase):
    def test_debug_template_files_exist(self) -> None:
        root = ROOT / 'src' / 'keil2cmake' / 'templates'
        self.assertTrue((root / 'cppcheck.cmake.j2').exists())
        self.assertTrue((root / 'openocd.cfg.in.j2').exists())
        self.assertTrue((root / 'launch.json.in.j2').exists())
        self.assertTrue((root / 'tasks.json.in.j2').exists())

    def test_launch_template_contains_required_fields(self) -> None:
        tpl = ROOT / 'src' / 'keil2cmake' / 'templates' / 'launch.json.in.j2'
        content = tpl.read_text(encoding='utf-8')
        self.assertIn('"servertype": "openocd"', content)
        self.assertIn('"configFiles"', content)
        self.assertIn('"serverpath"', content)
        self.assertIn('"gdbPath"', content)
        self.assertIn('"executable"', content)

    def test_tasks_template_contains_program_command(self) -> None:
        tpl = ROOT / 'src' / 'keil2cmake' / 'templates' / 'tasks.json.in.j2'
        content = tpl.read_text(encoding='utf-8')
        self.assertIn('"command": "@K2C_OPENOCD_PATH@"', content)
        self.assertIn('"-f"', content)
        self.assertIn('"-c"', content)
        self.assertIn('"program @K2C_DEBUG_EXECUTABLE@ verify reset exit"', content)
        self.assertIn('openocd.cfg', content)

    def test_openocd_cfg_template_contains_sources(self) -> None:
        tpl = ROOT / 'src' / 'keil2cmake' / 'templates' / 'openocd.cfg.in.j2'
        content = tpl.read_text(encoding='utf-8')
        self.assertIn('@K2C_OPENOCD_INTERFACE_LINE@', content)
        self.assertIn('@K2C_OPENOCD_TRANSPORT_LINE@', content)
        self.assertIn('@K2C_OPENOCD_TARGET_LINE@', content)
        self.assertIn('Optional transport list', content)
        self.assertNotIn('@K2C_OPENOCD_BOARD_LINE@', content)
        self.assertIn('OpenOCD config', content)

if __name__ == '__main__':
    unittest.main(verbosity=2)
