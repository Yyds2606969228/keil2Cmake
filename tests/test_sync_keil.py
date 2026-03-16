# -*- coding: utf-8 -*-

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from keil2cmake.cli import main as cli_main
from keil2cmake.keil.sync import sync_cmake_to_uvprojx


def _write_user_cmake(cmake_root: Path) -> None:
    user_dir = cmake_root / 'cmake' / 'user'
    user_dir.mkdir(parents=True, exist_ok=True)
    (cmake_root / 'src').mkdir(parents=True, exist_ok=True)
    (cmake_root / 'include').mkdir(parents=True, exist_ok=True)
    (cmake_root / 'src' / 'main.c').write_text('int main(void){return 0;}\n', encoding='utf-8')
    (cmake_root / 'src' / 'startup.s').write_text('.syntax unified\n', encoding='utf-8')
    (cmake_root / 'src' / 'extra.c').write_text('void extra(void){}\n', encoding='utf-8')

    content = '''
set(K2C_SOURCES
    "src/main.c"
    "src/startup.s"
)

set(K2C_INCLUDE_DIRS
    "include"
)

set(K2C_DEFINES
    USE_HAL_DRIVER
    STM32F103xB
)

set(K2C_KEIL_MISC_C_FLAGS "--target=arm-arm-none-eabi")
set(K2C_KEIL_MISC_ASM_FLAGS "--diag_suppress=1296")
set(K2C_KEIL_MISC_LD_FLAGS "--strict")
'''.strip()
    (user_dir / 'keil2cmake_user.cmake').write_text(content, encoding='utf-8')


def _write_uvprojx(path: Path, include_second_target: bool = False) -> None:
    project = '''
<Project>
  <Targets>
    <Target>
      <TargetName>TargetA</TargetName>
      <Groups>
        <Group>
          <GroupName>Src</GroupName>
          <Files>
            <File>
              <FileName>legacy.c</FileName>
              <FileType>1</FileType>
              <FilePath>legacy/legacy.c</FilePath>
            </File>
          </Files>
        </Group>
      </Groups>
      <TargetOption>
        <TargetArmAds>
          <Cads>
            <VariousControls>
              <IncludePath></IncludePath>
              <Define></Define>
              <MiscControls></MiscControls>
            </VariousControls>
          </Cads>
          <Aads>
            <VariousControls>
              <MiscControls></MiscControls>
            </VariousControls>
          </Aads>
          <LDads>
            <VariousControls>
              <MiscControls></MiscControls>
            </VariousControls>
          </LDads>
        </TargetArmAds>
      </TargetOption>
    </Target>
'''
    second = '''
    <Target>
      <TargetName>TargetB</TargetName>
      <Groups>
        <Group>
          <GroupName>BGroup</GroupName>
          <Files></Files>
        </Group>
      </Groups>
      <TargetOption>
        <TargetArmAds>
          <Cads><VariousControls><IncludePath></IncludePath><Define></Define><MiscControls></MiscControls></VariousControls></Cads>
          <Aads><VariousControls><MiscControls></MiscControls></VariousControls></Aads>
          <LDads><VariousControls><MiscControls></MiscControls></VariousControls></LDads>
        </TargetArmAds>
      </TargetOption>
    </Target>
'''
    ending = '''
  </Targets>
</Project>
'''
    text = project + (second if include_second_target else '') + ending
    path.write_text(text.strip(), encoding='utf-8')


class TestSyncKeil(unittest.TestCase):
    def test_sync_creates_group_and_updates_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            uvprojx = root / 'keil' / 'project.uvprojx'
            uvprojx.parent.mkdir(parents=True, exist_ok=True)
            _write_uvprojx(uvprojx)

            cmake_root = root / 'cmake_proj'
            _write_user_cmake(cmake_root)

            result = sync_cmake_to_uvprojx(str(uvprojx), str(cmake_root))
            self.assertEqual(result.added_count, 2)
            self.assertTrue(Path(result.backup_path).exists())

            tree = ET.parse(uvprojx)
            target = tree.getroot().find('./Targets/Target')
            self.assertIsNotNone(target)
            assert target is not None

            sync_group = None
            for group in target.findall('./Groups/Group'):
                if (group.findtext('GroupName') or '').strip() == 'K2C_Sync':
                    sync_group = group
                    break
            self.assertIsNotNone(sync_group)
            assert sync_group is not None

            files = sync_group.findall('./Files/File')
            self.assertEqual(len(files), 2)
            by_name = {
                (f.findtext('FileName') or '').strip(): (f.findtext('FileType') or '').strip()
                for f in files
            }
            self.assertEqual(by_name.get('main.c'), '1')
            self.assertEqual(by_name.get('startup.s'), '2')

            include_path = target.findtext('./TargetOption/TargetArmAds/Cads/VariousControls/IncludePath') or ''
            self.assertIn('cmake_proj/include', include_path.replace('\\', '/'))
            self.assertEqual(
                target.findtext('./TargetOption/TargetArmAds/Cads/VariousControls/Define'),
                'USE_HAL_DRIVER,STM32F103xB',
            )
            self.assertEqual(
                target.findtext('./TargetOption/TargetArmAds/Cads/VariousControls/MiscControls'),
                '--target=arm-arm-none-eabi',
            )

    def test_sync_prune_only_affects_sync_group(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            uvprojx = root / 'keil' / 'project.uvprojx'
            uvprojx.parent.mkdir(parents=True, exist_ok=True)
            _write_uvprojx(uvprojx)

            cmake_root = root / 'cmake_proj'
            _write_user_cmake(cmake_root)

            sync_cmake_to_uvprojx(str(uvprojx), str(cmake_root))

            tree = ET.parse(uvprojx)
            target = tree.getroot().find('./Targets/Target')
            assert target is not None
            sync_group = None
            for group in target.findall('./Groups/Group'):
                if (group.findtext('GroupName') or '').strip() == 'K2C_Sync':
                    sync_group = group
                    break
            assert sync_group is not None
            files_node = sync_group.find('./Files')
            assert files_node is not None

            extra = ET.SubElement(files_node, 'File')
            ET.SubElement(extra, 'FileName').text = 'legacy_sync.c'
            ET.SubElement(extra, 'FileType').text = '1'
            ET.SubElement(extra, 'FilePath').text = 'legacy/legacy_sync.c'
            if hasattr(ET, 'indent'):
                ET.indent(tree, space='  ')
            tree.write(uvprojx, encoding='utf-8', xml_declaration=True)

            pruned = sync_cmake_to_uvprojx(str(uvprojx), str(cmake_root), prune=True)
            self.assertGreaterEqual(pruned.removed_count, 1)

            tree = ET.parse(uvprojx)
            target = tree.getroot().find('./Targets/Target')
            assert target is not None

            src_group = None
            sync_group = None
            for group in target.findall('./Groups/Group'):
                name = (group.findtext('GroupName') or '').strip()
                if name == 'Src':
                    src_group = group
                if name == 'K2C_Sync':
                    sync_group = group
            self.assertIsNotNone(src_group)
            self.assertIsNotNone(sync_group)
            assert sync_group is not None

            sync_paths = [
                (node.findtext('FilePath') or '').replace('\\', '/')
                for node in sync_group.findall('./Files/File')
            ]
            self.assertTrue(all('legacy/legacy_sync.c' not in p for p in sync_paths))
            src_paths = [
                (node.findtext('FilePath') or '').replace('\\', '/')
                for node in src_group.findall('./Files/File')
            ]
            self.assertIn('legacy/legacy.c', src_paths)

    def test_sync_dry_run_does_not_modify_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            uvprojx = root / 'keil' / 'project.uvprojx'
            uvprojx.parent.mkdir(parents=True, exist_ok=True)
            _write_uvprojx(uvprojx)
            before = uvprojx.read_text(encoding='utf-8')

            cmake_root = root / 'cmake_proj'
            _write_user_cmake(cmake_root)

            result = sync_cmake_to_uvprojx(str(uvprojx), str(cmake_root), dry_run=True)
            self.assertTrue(result.dry_run)
            self.assertFalse(Path(str(uvprojx) + '.bak').exists())
            self.assertEqual(before, uvprojx.read_text(encoding='utf-8'))

    def test_sync_target_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            uvprojx = root / 'keil' / 'project.uvprojx'
            uvprojx.parent.mkdir(parents=True, exist_ok=True)
            _write_uvprojx(uvprojx, include_second_target=True)

            cmake_root = root / 'cmake_proj'
            _write_user_cmake(cmake_root)

            result = sync_cmake_to_uvprojx(str(uvprojx), str(cmake_root), target_name='TargetB')
            self.assertEqual(result.target_name, 'TargetB')

            tree = ET.parse(uvprojx)
            targets = tree.getroot().findall('./Targets/Target')
            self.assertEqual(len(targets), 2)

            target_a = targets[0]
            target_b = targets[1]
            names_a = [(node.findtext('GroupName') or '').strip() for node in target_a.findall('./Groups/Group')]
            names_b = [(node.findtext('GroupName') or '').strip() for node in target_b.findall('./Groups/Group')]
            self.assertNotIn('K2C_Sync', names_a)
            self.assertIn('K2C_Sync', names_b)

    def test_cli_sync_keil(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            uvprojx = root / 'keil' / 'project.uvprojx'
            uvprojx.parent.mkdir(parents=True, exist_ok=True)
            _write_uvprojx(uvprojx)

            cmake_root = root / 'cmake_proj'
            _write_user_cmake(cmake_root)

            ret = cli_main(
                [
                    'sync-keil',
                    '--uvprojx',
                    str(uvprojx),
                    '--cmake-root',
                    str(cmake_root),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue(Path(str(uvprojx) + '.bak').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
