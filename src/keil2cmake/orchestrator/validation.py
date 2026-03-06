from __future__ import annotations

from ..keil.config import get_cmake_path
from .models import CommandExecution


class ValidationExecutor:
    def run(
        self,
        *,
        stage: str,
        project_root: str,
        build_preset: str,
        configure_preset: str,
        execute_command,
    ) -> CommandExecution:
        if stage == 'build':
            return execute_command(
                [get_cmake_path() or 'cmake', '--build', '--preset', build_preset],
                project_root,
                execute=True,
            )
        if stage == 'configure':
            return execute_command(
                [get_cmake_path() or 'cmake', '--preset', configure_preset],
                project_root,
                execute=True,
            )
        raise ValueError(f'no validation executor is implemented for stage {stage!r}.')