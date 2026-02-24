# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass


def _shape_product(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        d = int(dim)
        if d <= 0:
            raise ValueError("MatMul requires known positive dimensions.")
        out *= d
    return out


def _full_row_major_strides(shape: list[int]) -> list[int]:
    rank = len(shape)
    strides = [0] * rank
    stride = 1
    for axis in range(rank - 1, -1, -1):
        dim = int(shape[axis])
        if dim <= 0:
            raise ValueError("MatMul requires known positive dimensions.")
        strides[axis] = stride
        stride *= dim
    return strides


def _broadcast_batch_shape(a_batch: list[int], b_batch: list[int], op_name: str) -> list[int]:
    out_rank = max(len(a_batch), len(b_batch))
    out_rev: list[int] = []
    for idx in range(out_rank):
        a_dim = int(a_batch[-1 - idx]) if idx < len(a_batch) else 1
        b_dim = int(b_batch[-1 - idx]) if idx < len(b_batch) else 1
        if a_dim <= 0 or b_dim <= 0:
            raise ValueError(f"{op_name} requires known positive dimensions.")
        if a_dim == b_dim or a_dim == 1 or b_dim == 1:
            out_rev.append(max(a_dim, b_dim))
        else:
            raise ValueError(f"{op_name} batch dimensions are not broadcast-compatible.")
    return list(reversed(out_rev))


def _broadcast_batch_strides(in_shape: list[int], out_batch_shape: list[int], op_name: str) -> list[int]:
    in_batch = [int(v) for v in in_shape[:-2]]
    in_rank = len(in_batch)
    out_rank = len(out_batch_shape)
    if in_rank > out_rank:
        raise ValueError(f"{op_name} batch rank mismatch.")
    full_strides = _full_row_major_strides([int(v) for v in in_shape])
    in_batch_strides = full_strides[:in_rank]
    pad = out_rank - in_rank
    out: list[int] = []
    for axis in range(out_rank):
        out_dim = int(out_batch_shape[axis])
        if out_dim <= 0:
            raise ValueError(f"{op_name} requires known positive dimensions.")
        if axis < pad:
            in_dim = 1
            stride = 0
        else:
            src_axis = axis - pad
            in_dim = int(in_batch[src_axis])
            stride = int(in_batch_strides[src_axis])
        if in_dim == out_dim:
            out.append(stride)
        elif in_dim == 1:
            out.append(0)
        else:
            raise ValueError(f"{op_name} batch dimensions are not broadcast-compatible.")
    return out


@dataclass(frozen=True)
class MatMulBatchPlan:
    m: int
    k: int
    n: int
    batch_shape: list[int]
    batch_size: int
    a_batch_strides: list[int]
    b_batch_strides: list[int]


def build_matmul_batch_plan(op_name: str, a_shape: list[int], b_shape: list[int]) -> MatMulBatchPlan:
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise ValueError(f"{op_name} requires rank >= 2 inputs.")
    m = int(a_shape[-2])
    k_a = int(a_shape[-1])
    k_b = int(b_shape[-2])
    n = int(b_shape[-1])
    for dim in (m, k_a, k_b, n):
        if dim <= 0:
            raise ValueError(f"{op_name} requires known positive dimensions.")
    if k_a != k_b:
        raise ValueError(f"{op_name} dimension mismatch.")
    batch_shape = _broadcast_batch_shape([int(v) for v in a_shape[:-2]], [int(v) for v in b_shape[:-2]], op_name)
    batch_size = _shape_product(batch_shape) if batch_shape else 1
    a_batch_strides = _broadcast_batch_strides([int(v) for v in a_shape], batch_shape, op_name)
    b_batch_strides = _broadcast_batch_strides([int(v) for v in b_shape], batch_shape, op_name)
    return MatMulBatchPlan(
        m=m,
        k=k_a,
        n=n,
        batch_shape=batch_shape,
        batch_size=batch_size,
        a_batch_strides=a_batch_strides,
        b_batch_strides=b_batch_strides,
    )
