# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

from typing import Tuple

import awkward as ak
import numpy as np

from .._spec_array_object import array, _unbox, _box
from .._import import device_namespace


def to_cf_contiguous(x: array) -> Tuple[array, array]:
    if x.ndim != 2:
        raise NotImplementedError

    (y,) = _unbox(x)

    return _box(type(x), ak.flatten(y)), _box(type(x), ak.num(y))


def from_cf_contiguous(content: array, counts: array) -> array:
    if content.ndim != 1 or counts.ndim != 1:
        raise NotImplementedError

    cont, cnts = _unbox(content, counts)

    return _box(type(content), ak.unflatten(cont, cnts))


def to_cf_indexed(x: array) -> Tuple[array, array]:
    if x.ndim != 2:
        raise NotImplementedError

    _, ns = device_namespace(x.device)
    (y,) = _unbox(x)

    index, _ = ak.broadcast_arrays(ns.arange(len(x), dtype=ns.int64), y)

    return _box(type(x), ak.flatten(y)), _box(type(x), ak.flatten(index))


def from_cf_indexed(content: array, index: array) -> array:
    if content.ndim != 1 or index.ndim != 1:
        raise NotImplementedError

    _, ns = device_namespace(content.device)
    cont, ind = _unbox(content, index)

    counts = ns.zeros(ak.max(ind) + 1, dtype=ns.int64)
    ns.add.at(counts, ns.asarray(ind), 1)

    return _box(type(content), ak.unflatten(cont[ns.argsort(ind)], counts))


__all__ = ["to_cf_contiguous", "from_cf_contiguous", "to_cf_indexed", "from_cf_indexed"]
