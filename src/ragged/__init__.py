# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
Ragged array module.

FIXME: needs more documentation!
"""

from __future__ import annotations

from ._helper_functions import regularise_to_float
from ._spec_array_object import array
from ._spec_constants import (
    e,
    inf,
    nan,
    newaxis,
    pi,
)
from ._spec_creation_functions import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)
from ._spec_data_type_functions import (
    astype,
    can_cast,
    finfo,
    iinfo,
    isdtype,
    result_type,
)
from ._spec_elementwise_functions import (  # pylint: disable=W0622
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    conj,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    imag,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    real,
    remainder,
    round,
    sign,
    sin,
    sinh,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    trunc,
)
from ._spec_indexing_functions import (
    take,
)
from ._spec_linear_algebra_functions import (
    matmul,
    matrix_transpose,
    tensordot,
    vecdot,
)
from ._spec_manipulation_functions import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    permute_dims,
    reshape,
    roll,
    squeeze,
    stack,
)
from ._spec_searching_functions import (
    argmax,
    argmin,
    nonzero,
    where,
)
from ._spec_set_functions import (
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
)
from ._spec_sorting_functions import (
    argsort,
    sort,
)
from ._spec_statistical_functions import (  # pylint: disable=W0622
    max,
    mean,
    min,
    prod,
    std,
    sum,
    var,
)
from ._spec_utility_functions import (  # pylint: disable=W0622
    all,
    any,
)

__array_api_version__ = "2022.12"

__all__ = [
    "__array_api_version__",
    # _spec_array_object
    "array",
    # _spec_constants
    "e",
    "inf",
    "nan",
    "newaxis",
    "pi",
    # _spec_creation_functions
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
    # _spec_data_type_functions
    "astype",
    "can_cast",
    "finfo",
    "iinfo",
    "isdtype",
    "result_type",
    # _spec_elementwise_functions
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "conj",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
    # _spec_indexing_functions
    "take",
    # _spec_linear_algebra_functions
    "matmul",
    "matrix_transpose",
    "tensordot",
    "vecdot",
    # _spec_manipulation_functions
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "permute_dims",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    # _spec_searching_functions
    "argmax",
    "argmin",
    "nonzero",
    "where",
    # _spec_set_functions
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    # _spec_sorting_functions
    "argsort",
    "sort",
    # _spec_statistical_functions
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "var",
    # _spec_utility_functions
    "all",
    "any",
    # _helper_functions
    "regularise_to_float",
]
