# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
"""

from __future__ import annotations

import pytest

import ragged


def test_existence():
    assert ragged.max is not None
    assert ragged.mean is not None
    assert ragged.min is not None
    assert ragged.prod is not None
    assert ragged.std is not None
    assert ragged.sum is not None
    assert ragged.var is not None


def test_max():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.max(data, axis=None).tolist() == 9.9
    assert (
        ragged.max(data, axis=0).tolist()
        == ragged.max(data, axis=-3).tolist()
        == [[3.3, 4.4, 2.2], [5.5], [6.6, 7.7, 8.8, 9.9]]
    )
    assert (
        ragged.max(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.max(data, axis=-2).tolist()
        == [[0, 1.1, 2.2], [], [6.6, 7.7, 8.8, 9.9]]
    )
    assert (
        ragged.max(data, axis=2).tolist()
        == ragged.max(data, axis=-1).tolist()
        == [[2.2, -ragged.inf], [], [4.4, 5.5, 9.9]]
    )
    assert (
        ragged.max(data, axis=(0, 1)).tolist()
        == ragged.max(data, axis=(1, 0)).tolist()
        == [6.6, 7.7, 8.8, 9.9]
    )
    assert (
        ragged.max(data, axis=(0, 2)).tolist()
        == ragged.max(data, axis=(2, 0)).tolist()
        == [4.4, 5.5, 9.9]
    )
    assert (
        ragged.max(data, axis=(1, 2)).tolist()
        == ragged.max(data, axis=(2, 1)).tolist()
        == [2.2, -ragged.inf, 9.9]
    )
    assert (
        ragged.max(data, axis=(0, 1, 2)).tolist()
        == ragged.max(data, axis=(-1, 0, 1)).tolist()
        == 9.9
    )


def test_mean():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.mean(data, axis=None).tolist() == pytest.approx(4.95)
    assert (
        ragged.mean(data, axis=0).tolist()  # type: ignore[comparison-overlap]
        == ragged.mean(data, axis=-3).tolist()
        == [
            pytest.approx([1.65, 2.75, 2.2]),
            pytest.approx([5.5]),
            pytest.approx([6.6, 7.7, 8.8, 9.9]),
        ]
    )
    assert (
        ragged.mean(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.mean(data, axis=-2).tolist()
        == [
            pytest.approx([0, 1.1, 2.2]),
            pytest.approx([]),
            pytest.approx([5.13333, 6.05, 8.8, 9.9]),
        ]
    )
    assert (
        ragged.mean(data, axis=2).tolist()  # type: ignore[comparison-overlap]
        == [
            pytest.approx([1.1, ragged.nan], nan_ok=True),
            pytest.approx([]),
            pytest.approx([3.85, 5.5, 8.25]),
        ]
    )
    assert (
        ragged.mean(data, axis=-1).tolist()  # type: ignore[comparison-overlap]
        == [
            pytest.approx([1.1, ragged.nan], nan_ok=True),
            pytest.approx([]),
            pytest.approx([3.85, 5.5, 8.25]),
        ]
    )
    assert (
        ragged.mean(data, axis=(0, 1)).tolist()
        == ragged.mean(data, axis=(1, 0)).tolist()
        == pytest.approx([3.85, 4.4, 5.5, 9.9])
    )
    assert (
        ragged.mean(data, axis=(0, 2)).tolist()
        == ragged.mean(data, axis=(2, 0)).tolist()
        == pytest.approx([2.2, 5.5, 8.25])
    )
    assert ragged.mean(data, axis=(1, 2)).tolist() == pytest.approx(
        [1.1, ragged.nan, 6.6], nan_ok=True
    )
    assert ragged.mean(data, axis=(2, 1)).tolist() == pytest.approx(
        [1.1, ragged.nan, 6.6], nan_ok=True
    )
    assert (
        ragged.mean(data, axis=(0, 1, 2)).tolist()
        == ragged.mean(data, axis=(-1, 0, 1)).tolist()
        == pytest.approx(4.95)
    )


def test_min():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.min(data, axis=None).tolist() == 0
    assert (
        ragged.min(data, axis=0).tolist()
        == ragged.min(data, axis=-3).tolist()
        == [[0, 1.1, 2.2], [5.5], [6.6, 7.7, 8.8, 9.9]]
    )
    assert (
        ragged.min(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.min(data, axis=-2).tolist()
        == [[0, 1.1, 2.2], [], [3.3, 4.4, 8.8, 9.9]]
    )
    assert (
        ragged.min(data, axis=2).tolist()
        == ragged.min(data, axis=-1).tolist()
        == [[0, ragged.inf], [], [3.3, 5.5, 6.6]]
    )
    assert (
        ragged.min(data, axis=(0, 1)).tolist()
        == ragged.min(data, axis=(1, 0)).tolist()
        == [0, 1.1, 2.2, 9.9]
    )
    assert (
        ragged.min(data, axis=(0, 2)).tolist()
        == ragged.min(data, axis=(2, 0)).tolist()
        == [0, 5.5, 6.6]
    )
    assert (
        ragged.min(data, axis=(1, 2)).tolist()
        == ragged.min(data, axis=(2, 1)).tolist()
        == [0, ragged.inf, 3.3]
    )
    assert (
        ragged.min(data, axis=(0, 1, 2)).tolist()
        == ragged.min(data, axis=(-1, 0, 1)).tolist()
        == 0
    )


def test_prod():
    data = ragged.array([[[2, 3, 5], []], [], [[7, 11], [13], [17, 19, 23, 27]]])
    assert ragged.prod(data, axis=None).tolist() == 6023507490
    assert (
        ragged.prod(data, axis=0).tolist()
        == ragged.prod(data, axis=-3).tolist()
        == [[14, 33, 5], [13], [17, 19, 23, 27]]
    )
    assert (
        ragged.prod(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.prod(data, axis=-2).tolist()
        == [[2, 3, 5], [], [1547, 209, 23, 27]]
    )
    assert (
        ragged.prod(data, axis=2).tolist()  # type: ignore[comparison-overlap]
        == ragged.prod(data, axis=-1).tolist()
        == [[30, 1], [], [77, 13, 200583]]
    )
    assert (
        ragged.prod(data, axis=(0, 1)).tolist()
        == ragged.prod(data, axis=(1, 0)).tolist()
        == [3094, 627, 115, 27]
    )
    assert (
        ragged.prod(data, axis=(0, 2)).tolist()
        == ragged.prod(data, axis=(2, 0)).tolist()
        == [2310, 13, 200583]
    )
    assert (
        ragged.prod(data, axis=(1, 2)).tolist()
        == ragged.prod(data, axis=(2, 1)).tolist()
        == [30, 1, 200783583]
    )
    assert (
        ragged.prod(data, axis=(0, 1, 2)).tolist()
        == ragged.prod(data, axis=(-1, 0, 1)).tolist()
        == 6023507490
    )


def test_std():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.std(data, axis=None).tolist() == pytest.approx(3.159509)
    assert (
        ragged.std(data, axis=0).tolist()  # type: ignore[comparison-overlap]
        == ragged.std(data, axis=-3).tolist()
        == [
            pytest.approx([1.65, 1.65, 0]),
            pytest.approx([0]),
            pytest.approx([0, 0, 0, 0]),
        ]
    )
    assert (
        ragged.std(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.std(data, axis=-2).tolist()
        == [
            pytest.approx([0, 0, 0]),
            pytest.approx([]),
            pytest.approx([1.37194, 1.65, 0, 0]),
        ]
    )
    assert (
        ragged.std(data, axis=2).tolist()  # type: ignore[comparison-overlap]
        == [
            pytest.approx([0.898146, ragged.nan], nan_ok=True),
            pytest.approx([]),
            pytest.approx([0.55, 0, 1.229837]),
        ]
    )
    assert (
        ragged.std(data, axis=-1).tolist()  # type: ignore[comparison-overlap]
        == [
            pytest.approx([0.898146, ragged.nan], nan_ok=True),
            pytest.approx([]),
            pytest.approx([0.55, 0, 1.229837]),
        ]
    )
    assert (
        ragged.std(data, axis=(0, 1, 2)).tolist()
        == ragged.std(data, axis=(-1, 0, 1)).tolist()
        == pytest.approx(3.159509)
    )


def test_sum():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.sum(data, axis=None).tolist() == pytest.approx(49.5)
    assert (
        ragged.sum(data, axis=0).tolist()  # type: ignore[comparison-overlap]
        == ragged.sum(data, axis=-3).tolist()
        == [
            pytest.approx([3.3, 5.5, 2.2]),
            pytest.approx([5.5]),
            pytest.approx([6.6, 7.7, 8.8, 9.9]),
        ]
    )
    assert (
        ragged.sum(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.sum(data, axis=-2).tolist()
        == [
            pytest.approx([0, 1.1, 2.2]),
            pytest.approx([]),
            pytest.approx([15.4, 12.1, 8.8, 9.9]),
        ]
    )
    assert (
        ragged.sum(data, axis=2).tolist()  # type: ignore[comparison-overlap]
        == ragged.sum(data, axis=-1).tolist()
        == [
            pytest.approx([3.3, 0]),
            pytest.approx([]),
            pytest.approx([7.7, 5.5, 33.0]),
        ]
    )
    assert (
        ragged.sum(data, axis=(0, 1)).tolist()
        == ragged.sum(data, axis=(1, 0)).tolist()
        == pytest.approx([15.4, 13.2, 11.0, 9.9])
    )
    assert (
        ragged.sum(data, axis=(0, 2)).tolist()
        == ragged.sum(data, axis=(2, 0)).tolist()
        == pytest.approx([11.0, 5.5, 33.0])
    )
    assert (
        ragged.sum(data, axis=(1, 2)).tolist()
        == ragged.sum(data, axis=(2, 1)).tolist()
        == pytest.approx([3.3, 0, 46.2])
    )
    assert (
        ragged.sum(data, axis=(0, 1, 2)).tolist()
        == ragged.sum(data, axis=(-1, 0, 1)).tolist()
        == pytest.approx(49.5)
    )


def test_var():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.var(data, axis=None).tolist() == pytest.approx(9.9825)
    assert (
        ragged.var(data, axis=0).tolist()  # type: ignore[comparison-overlap]
        == ragged.var(data, axis=-3).tolist()
        == [
            pytest.approx([2.7225, 2.7225, 0]),
            pytest.approx([0]),
            pytest.approx([0, 0, 0, 0]),
        ]
    )
    assert (
        ragged.var(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.var(data, axis=-2).tolist()
        == [
            pytest.approx([0, 0, 0]),
            pytest.approx([]),
            pytest.approx([1.88222222, 2.7225, 0, 0]),
        ]
    )
    assert (
        ragged.var(data, axis=2).tolist()  # type: ignore[comparison-overlap]
        == [
            pytest.approx([0.80666667, ragged.nan], nan_ok=True),
            pytest.approx([]),
            pytest.approx([0.3025, 0, 1.5125]),
        ]
    )
    assert (
        ragged.var(data, axis=-1).tolist()  # type: ignore[comparison-overlap]
        == [
            pytest.approx([0.80666667, ragged.nan], nan_ok=True),
            pytest.approx([]),
            pytest.approx([0.3025, 0, 1.5125]),
        ]
    )
    assert (
        ragged.var(data, axis=(0, 1, 2)).tolist()
        == ragged.var(data, axis=(-1, 0, 1)).tolist()
        == pytest.approx(9.9825)
    )
