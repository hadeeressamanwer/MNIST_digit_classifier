import numpy as np


def initialize_parameters_zeros(layers_dims):

    parameters = {}
    L = len(layers_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layers_dims[l], 1))
    return parameters


def initialize_parameters_random(layers_dims):
    parameters = {}
    L = len(layers_dims)  # integer representing the number of layers
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


