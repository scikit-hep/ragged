import numpy as np
import awkward as ak
import ragged 

def unique_values1d(arr):
    if not isinstance(arr, ragged.array):
        print("Input is not a ragged array")
        
    if arr.ndim != 1:
        print("Input is not a 1D array")

    if len(arr)==1:
        return arr

    arr_np = ak.to_numpy(arr)
    unique_np = np.unique(arr_np)
    unique_arr = ak.from_numpy(unique_np)

    return unique_arr
