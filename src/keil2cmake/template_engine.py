# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).resolve().parent / 'templates'

_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(name: str, context: dict[str, Any]) -> str:
    return _ENV.get_template(name).render(**context)


def write_template(name: str, context: dict[str, Any], output_path: str, encoding: str = 'utf-8') -> None:
    content = render_template(name, context)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding=encoding)


def render_at_template(name: str, context: dict[str, Any]) -> str:
    content = (_TEMPLATE_DIR / name).read_text(encoding='utf-8')
    for key, value in context.items():
        content = content.replace(f"@{key}@", str(value))
    return content


def write_at_template(name: str, context: dict[str, Any], output_path: str, encoding: str = 'utf-8') -> None:
    content = render_at_template(name, context)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding=encoding)
