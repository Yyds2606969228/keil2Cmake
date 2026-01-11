# -*- coding: utf-8 -*-

import os

from keil2cmake_common import ensure_dir, format_cmake_list, norm_path, SUPPORTED_COMPILERS
from keil.device import detect_cpu_architecture
from keil.config import get_cmake_min_version
from i18n import t


def _relativize_paths(paths: list[str], project_root: str, uvprojx_dir: str | None) -> list[str]:
    uv_base = uvprojx_dir or project_root
    out: list[str] = []
    for p in paths or []:
        raw = str(p).strip()
        if not raw:
            continue

        # Resolve relative paths as Keil does: relative to the .uvprojx directory.
        if os.path.isabs(raw):
            abs_path = os.path.normpath(raw)
        else:
            abs_path = os.path.normpath(os.path.join(uv_base, raw))

        # Prefer a nice relative path in generated CMake (more portable).
        try:
            rel = os.path.relpath(abs_path, project_root)
            out.append(norm_path(rel))
        except Exception:
            out.append(norm_path(abs_path))
    return out


def generate_cmake_structure(project_data: dict, project_root: str) -> None:
    """Generate layered CMake structure under project_root."""
    cmake_min_version = get_cmake_min_version()
    
    # Map Keil Optim value to compiler-specific optimization level
    keil_optim = project_data.get('keil_optim', '0')
    use_armclang = project_data.get('use_armclang', False)
    
    # Keil Optim mapping depends on compiler type
    if use_armclang:
        # ARMCLANG (Compiler 6) mapping
        optim_map = {
            '0': '0',   # -O0
            '1': '1',   # -O1
            '2': '2',   # -O2
            '3': '3',   # -O3
            '4': '1',   # Keil default for AC6
            '11': 'z',  # -Oz (minimize size)
        }
    else:
        # ARMCC (Compiler 5) mapping
        optim_map = {
            '0': '0',   # -O0
            '1': '1',   # -O1
            '2': '2',   # -O2
            '3': '3',   # -O3
            '4': '1',   # Keil default for AC5
            '11': 's',  # -Ospace (minimize size)
        }
    
    opt_level = optim_map.get(keil_optim, '0')
    project_data['opt_level'] = opt_level

    user_dir = os.path.join(project_root, 'cmake', 'user')
    ensure_dir(user_dir)

    uvprojx_dir = project_data.get('uvprojx_dir')
    gen_sources = _relativize_paths(project_data.get('source_files', []), project_root, uvprojx_dir)
    gen_includes = _relativize_paths(project_data.get('include_paths', []), project_root, uvprojx_dir)
    gen_defines = [d.strip() for d in project_data['defines'] if str(d).strip()]

    defines_block = "\n    ".join(gen_defines)
    keil_misc_c_flags = project_data['c_flags'].replace('"', '\\"')
    keil_misc_asm_flags = project_data['asm_flags'].replace('"', '\\"')
    keil_misc_ld_flags = project_data['ld_flags'].replace('"', '\\"')

    user_cmake = f'''{t('gen.user.header.title')}
{t('gen.user.header.safe')}
{t('gen.user.header.no_overwrite')}

set(K2C_PROJECT_NAME "{project_data['project_name']}")
set(K2C_DEVICE "{project_data['device']}")
set(K2C_CPU_ARCH "{detect_cpu_architecture(project_data['device'])}")

{t('gen.user.defaults')}
set(K2C_DEFAULT_COMPILER "{'armclang' if project_data['use_armclang'] else 'armcc'}")
set(K2C_DEFAULT_OPTIMIZE_LEVEL "{project_data['opt_level']}")

{t('gen.user.optimize')}
set(K2C_OPTIMIZE_LEVEL "" CACHE STRING "Force optimize level (0/1/2/3/s). Empty = use Keil default")

{t('gen.user.linker')}
set(K2C_LINKER_SCRIPT_SCT "" CACHE FILEPATH "Scatter file for armcc/armclang")
set(K2C_LINKER_SCRIPT_LD  "" CACHE FILEPATH "Linker script for armgcc")

set(K2C_SOURCES
    {format_cmake_list(gen_sources)}
)

set(K2C_INCLUDE_DIRS
    {format_cmake_list(gen_includes)}
)

set(K2C_DEFINES
    {defines_block}
)

set(K2C_KEIL_MISC_C_FLAGS "{keil_misc_c_flags}")
set(K2C_KEIL_MISC_ASM_FLAGS "{keil_misc_asm_flags}")
set(K2C_KEIL_MISC_LD_FLAGS "{keil_misc_ld_flags}")
'''

    user_cmake_path = os.path.join(user_dir, 'keil2cmake_user.cmake')
    if not os.path.exists(user_cmake_path):
        with open(user_cmake_path, 'w', encoding='utf-8-sig') as f:
            f.write(user_cmake)

    top_level = f'''cmake_minimum_required(VERSION {cmake_min_version})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project({project_data['project_name']} LANGUAGES C ASM)

include(${{CMAKE_SOURCE_DIR}}/cmake/user/keil2cmake_user.cmake)

# K2C_COMPILER comes from toolchain/preset
message(STATUS "K2C compiler: ${{K2C_COMPILER}}")
message(STATUS "K2C optimize: ${{K2C_OPTIMIZE_LEVEL}}")
message(STATUS "K2C device:   ${{K2C_DEVICE}}")
message(STATUS "K2C cpu:      ${{K2C_CPU_ARCH}}")

if(NOT K2C_OPTIMIZE_LEVEL OR K2C_OPTIMIZE_LEVEL STREQUAL "")
    set(K2C_OPTIMIZE_LEVEL "${{K2C_DEFAULT_OPTIMIZE_LEVEL}}")
endif()

add_executable(${{PROJECT_NAME}}
    ${{K2C_SOURCES}}
)

target_include_directories(${{PROJECT_NAME}} PRIVATE
    ${{K2C_INCLUDE_DIRS}}
)

target_compile_definitions(${{PROJECT_NAME}} PRIVATE
    ${{K2C_DEFINES}}
)

# Optimization flags differ across compilers
# Important: do NOT apply -O* to ASM (ARMASM rejects -O0).
set(_K2C_OPT_FLAG "")
if(K2C_COMPILER STREQUAL "armcc")
    if(K2C_OPTIMIZE_LEVEL STREQUAL "s")
        set(_K2C_OPT_FLAG "-Ospace")  # ARMCC uses -Ospace for size optimization
    else()
        set(_K2C_OPT_FLAG "-O${{K2C_OPTIMIZE_LEVEL}}")
    endif()
elseif(K2C_COMPILER STREQUAL "armclang")
    if(K2C_OPTIMIZE_LEVEL STREQUAL "z")
        set(_K2C_OPT_FLAG "-Oz")  # ARMCLANG uses -Oz for minimal size
    else()
        set(_K2C_OPT_FLAG "-O${{K2C_OPTIMIZE_LEVEL}}")
    endif()
else()  # armgcc
    if(K2C_OPTIMIZE_LEVEL STREQUAL "s")
        set(_K2C_OPT_FLAG "-Os")  # GCC uses -Os
    else()
        set(_K2C_OPT_FLAG "-O${{K2C_OPTIMIZE_LEVEL}}")
    endif()
endif()

target_compile_options(${{PROJECT_NAME}} PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${{_K2C_OPT_FLAG}}>
    $<$<COMPILE_LANGUAGE:CXX>:${{_K2C_OPT_FLAG}}>
)

# Linker options
if(K2C_COMPILER STREQUAL "armcc" OR K2C_COMPILER STREQUAL "armclang")
    if(K2C_LINKER_SCRIPT_SCT STREQUAL "")
        message(FATAL_ERROR "K2C_LINKER_SCRIPT_SCT not set")
    endif()
    target_link_options(${{PROJECT_NAME}} PRIVATE "--scatter=${{K2C_LINKER_SCRIPT_SCT}}")

    if(DEFINED CMAKE_FROMELF)
        add_custom_command(TARGET ${{PROJECT_NAME}} POST_BUILD
            COMMAND ${{CMAKE_FROMELF}} --i32combined --output="${{CMAKE_CURRENT_BINARY_DIR}}/${{PROJECT_NAME}}.hex" "$<TARGET_FILE:${{PROJECT_NAME}}>"
            COMMENT "Generating HEX file"
        )
        add_custom_command(TARGET ${{PROJECT_NAME}} POST_BUILD
            COMMAND ${{CMAKE_FROMELF}} --bin --output="${{CMAKE_CURRENT_BINARY_DIR}}/${{PROJECT_NAME}}.bin" "$<TARGET_FILE:${{PROJECT_NAME}}>"
            COMMENT "Generating BIN file"
        )
    endif()
elseif(K2C_COMPILER STREQUAL "armgcc")
    if(K2C_LINKER_SCRIPT_LD STREQUAL "")
        message(FATAL_ERROR "K2C_LINKER_SCRIPT_LD not set")
    endif()
    target_link_options(${{PROJECT_NAME}} PRIVATE
        "-T${{K2C_LINKER_SCRIPT_LD}}"
        "-Wl,-Map=${{CMAKE_CURRENT_BINARY_DIR}}/${{PROJECT_NAME}}.map"
        "-Wl,--gc-sections"
    )

    if(DEFINED CMAKE_OBJCOPY)
        add_custom_command(TARGET ${{PROJECT_NAME}} POST_BUILD
            COMMAND ${{CMAKE_OBJCOPY}} -O ihex "$<TARGET_FILE:${{PROJECT_NAME}}>" "${{CMAKE_CURRENT_BINARY_DIR}}/${{PROJECT_NAME}}.hex"
            COMMENT "Generating HEX file"
        )
        add_custom_command(TARGET ${{PROJECT_NAME}} POST_BUILD
            COMMAND ${{CMAKE_OBJCOPY}} -O binary "$<TARGET_FILE:${{PROJECT_NAME}}>" "${{CMAKE_CURRENT_BINARY_DIR}}/${{PROJECT_NAME}}.bin"
            COMMENT "Generating BIN file"
        )
    endif()
endif()

# Custom target to show available CMake options
add_custom_target(show-options
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    COMMAND ${{CMAKE_COMMAND}} -E echo "========================================="
    COMMAND ${{CMAKE_COMMAND}} -E echo "Keil2CMake Project Build Options"
    COMMAND ${{CMAKE_COMMAND}} -E echo "========================================="
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    COMMAND ${{CMAKE_COMMAND}} -E echo "Available CMake Presets:"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake --preset keil2cmake         (use default compiler)"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake --preset keil2cmake-armcc"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake --preset keil2cmake-armclang"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake --preset keil2cmake-armgcc"
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    COMMAND ${{CMAKE_COMMAND}} -E echo "Build Commands:"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake --build --preset keil2cmake"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake --build build --config Debug"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake --build build --config Release"
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    COMMAND ${{CMAKE_COMMAND}} -E echo "Cache Variables (set via -D):"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  K2C_COMPILER=<armcc|armclang|armgcc>"
    COMMAND ${{CMAKE_COMMAND}} -E echo "    Default: ${{K2C_DEFAULT_COMPILER}}"
    COMMAND ${{CMAKE_COMMAND}} -E echo "    Override compiler selection"
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    COMMAND ${{CMAKE_COMMAND}} -E echo "  K2C_OPTIMIZE_LEVEL=<0|1|2|3|s|z>"
    COMMAND ${{CMAKE_COMMAND}} -E echo "    Default: ${{K2C_DEFAULT_OPTIMIZE_LEVEL}} (from Keil project)"
    COMMAND ${{CMAKE_COMMAND}} -E echo "    Override optimization level"
    COMMAND ${{CMAKE_COMMAND}} -E echo "    0=none, 1=O1, 2=O2, 3=O3, s=size(armcc), z=size(armclang)"
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    COMMAND ${{CMAKE_COMMAND}} -E echo "  K2C_LINKER_SCRIPT_SCT=<path>"
    COMMAND ${{CMAKE_COMMAND}} -E echo "    Override scatter file (for armcc/armclang)"
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    COMMAND ${{CMAKE_COMMAND}} -E echo "  K2C_LINKER_SCRIPT_LD=<path>"
    COMMAND ${{CMAKE_COMMAND}} -E echo "    Override linker script (for armgcc)"
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    COMMAND ${{CMAKE_COMMAND}} -E echo "Examples:"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake -B build -DK2C_COMPILER=armgcc"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake -B build -DK2C_OPTIMIZE_LEVEL=3"
    COMMAND ${{CMAKE_COMMAND}} -E echo "  cmake --build build --target show-options"
    COMMAND ${{CMAKE_COMMAND}} -E echo ""
    VERBATIM
)
'''

    with open(os.path.join(project_root, 'CMakeLists.txt'), 'w', encoding='utf-8') as f:
        f.write(top_level)


def clean_generated(project_root: str) -> int:
    """Remove keil2cmake generated artifacts under project_root (safe list)."""
    removed = 0

    root_files = ['CMakeLists.txt', 'CMakePresets.json', '.clangd']
    for name in root_files:
        path = os.path.join(project_root, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass

    cmake_dir = os.path.join(project_root, 'cmake')
    internal_dir = os.path.join(cmake_dir, 'internal')
    user_dir = os.path.join(cmake_dir, 'user')

    known = [
        os.path.join(internal_dir, 'toolchain.cmake'),
        os.path.join(internal_dir, 'keil2cmake_default.sct'),
        os.path.join(internal_dir, 'keil2cmake_default.ld'),

        # Legacy internal layout (older generator versions)
        os.path.join(internal_dir, 'armcc', 'toolchain.cmake'),
        os.path.join(internal_dir, 'armclang', 'toolchain.cmake'),
        os.path.join(internal_dir, 'armgcc', 'toolchain.cmake'),
        os.path.join(internal_dir, 'keil2cmake_generated.cmake'),
        os.path.join(internal_dir, 'common', 'keil2cmake_generated.cmake'),

        os.path.join(user_dir, 'keil2cmake_user.cmake'),

        os.path.join(user_dir, 'common', 'keil2cmake_project.cmake'),
        os.path.join(user_dir, 'common', 'keil2cmake_user.cmake'),

        # Legacy user layout (older generator versions)
        os.path.join(user_dir, 'common', 'Template.sct'),
        os.path.join(user_dir, 'common', 'Template.ld'),
        # Legacy user root file (older generator versions)
        os.path.join(user_dir, 'keil2cmake_user.cmake'),
        os.path.join(user_dir, 'armcc', 'keil2cmake_user.cmake'),
        os.path.join(user_dir, 'armclang', 'keil2cmake_user.cmake'),
        os.path.join(user_dir, 'armgcc', 'keil2cmake_user.cmake'),
        os.path.join(user_dir, 'armcc', 'Template.sct'),
        os.path.join(user_dir, 'armclang', 'Template.sct'),
        os.path.join(user_dir, 'armgcc', 'Template.ld'),
    ]
    for path in known:
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass

    # Best-effort: remove old internal subdirs if empty (from older generator versions)
    for sub in ('armcc', 'armclang', 'armgcc', 'common'):
        d = os.path.join(internal_dir, sub)
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass

    # Best-effort: remove old user subdirs if empty (from older generator versions)
    for sub in ('armcc', 'armclang', 'armgcc', 'common'):
        d = os.path.join(user_dir, sub)
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass

    if removed:
        print(t('clean.done', count=removed))
    else:
        print(t('clean.none'))
    return 0
