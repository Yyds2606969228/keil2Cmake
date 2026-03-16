# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TinyML ONNX -> C code generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s onnx --model model.onnx --output ./out --weights flash --emit c
  %(prog)s onnx --model model.onnx --output ./out --emit lib --toolchain-bin C:/gcc-arm/bin
""",
    )

    sub = parser.add_subparsers(dest="cmd")

    onnx_p = sub.add_parser(
        "onnx",
        help="Generate TinyML artifacts from ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    onnx_p.add_argument("--model", required=True, help="Path to ONNX model")
    onnx_p.add_argument("--output", default="./onnx-for-mcu", help="Output root directory")
    onnx_p.add_argument("--weights", default="flash", choices=["flash", "ram"], help="Weight storage location")
    onnx_p.add_argument("--emit", default="c", choices=["c", "lib"], help="Emit C source or static library")
    onnx_p.add_argument(
        "--toolchain-bin",
        default="",
        help="Directory or file path used to locate arm-none-eabi-gcc/ar (only used when --emit lib)",
    )
    onnx_p.add_argument(
        "--no-strict-validation",
        dest="strict_validation",
        action="store_false",
        help="Allow generation when consistency validation is skipped.",
    )
    onnx_p.set_defaults(strict_validation=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd != "onnx":
        parser.print_help()
        return 2

    from .project import generate_tinyml_project

    if not os.path.exists(args.model):
        print(f"[error] file not found: {args.model}")
        return 1

    result = generate_tinyml_project(
        args.model,
        args.output,
        args.weights,
        args.emit,
        toolchain_bin=args.toolchain_bin,
        strict_validation=bool(args.strict_validation),
    )

    print("[ok] generated")
    print(f"  model: {result['model_name']}")
    print(f"  output: {os.path.abspath(result['project_dir'])}")
    print(f"  weights: {result['weights']}")
    print(f"  emit: {args.emit}")
    print(f"  header: {result['header']}")
    print(f"  source: {result['source']}")
    print(f"  manifest: {result['manifest']}")
    if result.get("library"):
        print(f"  library: {result['library']}")
    validation = result.get("validation")
    if validation is not None:
        status = getattr(validation, "status", "")
        reason = getattr(validation, "reason", "") or ""
        engine = getattr(validation, "engine", "") or ""
        extras = "; ".join([x for x in (engine, reason) if x])
        if extras:
            print(f"  validation: {status} ({extras})")
        else:
            print(f"  validation: {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

