import awkward as ak
import ragged
import numpy as np


def unique_counts(arr):
    if not isinstance(arr, ragged.array):
        print("Input is not a ragged array")
        return None
    if len(arr)==1:
        return [(arr[0], 1)]

    arr_flat=ak.ravel(arr)
    arr_np = ak.to_numpy(arr_flat)
    unique_elements, counts = np.unique(arr_np, return_counts=True)
    unique_arr_counts = list(zip(unique_elements, counts))

    return unique_arr_counts