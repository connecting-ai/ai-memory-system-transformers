import numpy as np

def linear_quantize(arr, num_bits=8):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_range = arr_max - arr_min

    scale_factor = (2 ** num_bits - 1) / arr_range
    quantized_arr = np.round((arr - arr_min) * scale_factor).astype(np.uint8)

    return quantized_arr, arr_min, arr_max, scale_factor

#We don't really need the dequantizer as for cosine similarity we can directly use the linear quantized version
def dequantize(quantized_arr, arr_min, scale_factor):
    dequantized_arr = (quantized_arr / scale_factor) + arr_min
    return dequantized_arr

# Assuming vector_query is your original embedding vector
num_bits = 8  # You can adjust the number of bits as needed for your desired storage size vs. fidelity trade-off