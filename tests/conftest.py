# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

import reprlib

import pytest

import ragged


@pytest.fixture(scope="session", autouse=True)
def _patch_reprlib():
    if not hasattr(reprlib.Repr, "repr1_original"):

        def repr1(self, x, level):
            if isinstance(x, ragged.array):
                return self.repr_instance(x, level)
            return self.repr1_original(x, level)

        reprlib.Repr.repr1_original = reprlib.Repr.repr1  # type: ignore[attr-defined]
        reprlib.Repr.repr1 = repr1  # type: ignore[method-assign]
