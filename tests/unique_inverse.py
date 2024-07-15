def unique_inverse(arr):
    if not isinstance(arr, ragged.array):
        print("Input is not a ragged array")
        
    if len(arr) == 1:
        return arr, np.array([0])
        
    arr_list=ak.ravel(arr)
    arr_np = ak.to_numpy(arr_list)
    unique_elements, first_indices = np.unique(arr_np, return_index=True)
    
    sorted_indices = np.argsort(first_indices)
    unique_elements = unique_elements[sorted_indices]
    first_indices = first_indices[sorted_indices]
    
    unique_arr = ak.from_numpy(unique_elements)
    unique_indices_arr = ak.from_numpy(first_indices)

    return unique_arr, unique_indices_arr