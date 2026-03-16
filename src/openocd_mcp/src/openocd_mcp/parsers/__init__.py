"""Parsers and symbol resolvers."""

from .elf_resolver import ELFResolver
from .svd_resolver import RegisterInfo, SVDResolver

__all__ = ["ELFResolver", "SVDResolver", "RegisterInfo"]
