# -*- coding: utf-8 -*-

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from i18n import set_language, t


class TestI18n(unittest.TestCase):
    def test_translate_basic(self) -> None:
        set_language('en')
        self.assertEqual(t('cli.show_config.toolchains'), 'Toolchain configuration:')
        set_language('zh')
        self.assertEqual(t('cli.show_config.toolchains'), '当前工具链配置:')


class TestConfigLangAndCli(unittest.TestCase):
    def test_config_language_default_and_cli_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, 'path.cfg')
            os.environ['KEIL2CMAKE_CONFIG_PATH'] = cfg_path

            # Import here so it respects env override in the same process.
            from keil import config as kcfg
            from keil2cmake_cli import main

            # default is zh
            self.assertEqual(kcfg.get_language(), 'zh')

            # persist en into config
            cfg = kcfg.load_config()
            cfg['GENERAL']['LANGUAGE'] = 'en'
            kcfg.save_config(cfg)
            self.assertEqual(kcfg.get_language(), 'en')

            # CLI override should win
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(['--show-config', '--lang', 'zh'])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn('当前工具链配置:', out)


class TestUvprojxAndGeneration(unittest.TestCase):
    def test_mdk_arm_output_root_and_relativized_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = os.path.join(td, 'demo')
            mdk_dir = os.path.join(project_root, 'MDK-ARM')
            os.makedirs(mdk_dir, exist_ok=True)

            # Create a dummy source file referenced relatively from the uvprojx
            src_rel = os.path.join('..', 'Core', 'startup.s')
            src_abs = os.path.join(project_root, 'Core', 'startup.s')
            os.makedirs(os.path.dirname(src_abs), exist_ok=True)
            with open(src_abs, 'w', encoding='utf-8') as f:
                f.write('; dummy asm\n')

            uvprojx_path = os.path.join(mdk_dir, 'qr.uvprojx')
            uvprojx_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Project>
  <Targets>
    <Target>
      <TargetName>qr</TargetName>
      <TargetOption>
        <TargetCommonOption>
          <Device>STM32F103C8</Device>
          <OutputDirectory>build/</OutputDirectory>
        </TargetCommonOption>
        <TargetArmAds>
          <UseArmClang>0</UseArmClang>
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
      </TargetOption>
      <Groups>
        <Group>
          <GroupName>Startup</GroupName>
          <Files>
            <File>
              <FilePath>{src_rel}</FilePath>
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

            # compute_project_root should go to parent of MDK-ARM
            from keil2cmake_cli import compute_project_root
            out_root = compute_project_root('.', uvprojx_path)
            self.assertEqual(os.path.normpath(out_root), os.path.normpath(project_root))

            # parse + generate should relativize against project root
            from keil.uvprojx import parse_uvprojx
            from project_gen import generate_cmake_structure

            set_language('en')
            data = parse_uvprojx(uvprojx_path)
            self.assertTrue(data.get('uvprojx_dir', '').endswith('MDK-ARM'))

            generate_cmake_structure(data, project_root)

            gen_path = os.path.join(project_root, 'cmake', 'user', 'keil2cmake_user.cmake')
            self.assertTrue(os.path.exists(gen_path))
            with open(gen_path, 'r', encoding='utf-8') as f:
                gen = f.read()

            # The source should be written as a path relative to project_root.
            self.assertIn('Core/startup.s', gen.replace('\\', '/'))

            # Ensure we keep ASM optimization isolation in top-level CMakeLists.
            cmakelists = os.path.join(project_root, 'CMakeLists.txt')
            with open(cmakelists, 'r', encoding='utf-8') as f:
                top = f.read()
            self.assertIn('$<$<COMPILE_LANGUAGE:C>:', top)
            self.assertIn('$<$<COMPILE_LANGUAGE:CXX>:', top)
            self.assertNotIn('K2C_EXTRA_SOURCES', top)
            self.assertNotIn('K2C_EXTRA_INCLUDE_DIRS', top)
            self.assertNotIn('K2C_EXTRA_DEFINES', top)


if __name__ == '__main__':
    unittest.main(verbosity=2)
