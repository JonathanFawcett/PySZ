import numpy as np

def verify_order_of_magnitude(result, desired, req_order=1):
    order_of_mag = np.log10(abs(result - desired)) 
    return np.all(order_of_mag <= req_order)