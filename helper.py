# Arora, Priyank
# 1001-553-349
# 2018-09-06
# Assignment-01-02
import numpy as np
# This module calculates the activation function
def calculate_activation_function(weight,bias,input_array,type='Sigmoid'):
    net_value = weight * input_array + bias
    if type == 'Sigmoid':
        activation = 1.0 / (1 + np.exp(-net_value))
    elif type == "Linear":
        activation = net_value
    elif type == "Hyperbolic Tangent":
        activation = np.tanh(net_value)
    elif type == "Positive linear (RELU)":
        activation = np.maximum(net_value,0)
    return activation
