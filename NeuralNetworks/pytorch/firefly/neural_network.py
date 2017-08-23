# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import pickle
import os
import signal


class GracefulInterruptHandler(object):
    """ This interrupt handler is used to allow the user to stop training by
    hitting control-c. """

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)
        def handler(signum, frame):
            self.release()
            self.interrupted = True
        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


def create_network(dimensions, activation_functions):
    """ The dimensions argument is a list that specifies the number of
    inputs to the network and the number of units in each layer.  The
    number of inputs to the network is dimensions[0] and the number of
    units in the nth layer is dimensions[n].  The activation_functions
    argument is a list of functions to be used as the activation fuction
    for each layer of the network. """
    assert len(dimensions) == len(activation_functions) + 1
    ops = []
    for i in range(len(dimensions) - 1):
        ops.append(torch.nn.Linear(dimensions[i], dimensions[i+1]))
        ops.append(activation_functions[i])
    network = torch.nn.Sequential(*ops)
    return network


def set_parameters(network, parameter_arg):
    """ If the parameters argument is a dictionary then set the network
    parameters using the values given. If the argument is a filename then read
    the parameters from the file. """
    if isinstance(parameter_arg, str):
        filename = parameter_arg
        with open(filename, 'r') as file_handle:
            parameters = pickle.load(file_handle)
    elif isinstance(parameter_arg, dict):
        parameters = parameter_arg
        if 'weights' in parameters.keys():
            weights = parameters['weights']
        if 'biases' in parameters.keys():
            bias = parameters['biases']
    else:
        raise TypeError("set_parameters: unknown type for parameter_arg")
    assert len(weights) == len(bias)
    assert len(weights) == sum([isinstance(child, torch.nn.Linear)
                                for child in network.children()])
    i = 0
    for child in network.children():
        if isinstance(child, torch.nn.Linear):
            child.weight.data = torch.Tensor(weights[i])
            child.bias.data = torch.Tensor(bias[i])
            i = i + 1


def to_array(argument):
    """ Make an array using data from a torch tensor or variable. """
    if isinstance(argument, torch.Tensor):
        tensor = argument 
    elif isinstance(argument, torch.autograd.Variable):
        tensor = argument.data
    else:
        raise TypeError("to_array: unknown argument type")
    num_elements = int(np.prod(tensor.size()))
    flat_tensor = torch.Tensor(tensor).resize_(num_elements)
    flat_data = [flat_tensor[i] for i in range(num_elements)]
    return np.array(flat_data).reshape(tensor.size())


def get_parameters(network):
    """ Return network weights and biases in a Variable containing a 1D tensor
    for regularization. """
    parameters = None
    for child in network.children():
        if isinstance(child, torch.nn.Linear):
            num_weights = int(np.prod(child.weight.size()))
            flat_weights = child.weight.resize(num_weights)
            if isinstance(parameters, Variable):
                parameters = torch.cat([parameters, flat_weights], dim=0)
            else:
                parameters = flat_weights
            num_biases = int(np.prod(child.bias.size()))
            flat_biases = child.bias.resize(num_biases)
            parameters = torch.cat([parameters, flat_biases], dim=0)
    return parameters


def save_parameters(network, filename):
    """ Save the network's weights and biases. """
    if not os.path.isfile(filename):
        # Don't overwrite existing files.
        weights = []
        biases = []
        for child in network.children():
            if isinstance(child, torch.nn.Linear):
                weights.append(child.weight)
                biases.append(child.bias)
        parameters = (weights, biases)
        with open(filename, 'w') as file_handle:
            pickle.dump(parameters, file_handle)
    else:
        # Let the user know the weights were not saved.
        raise IOError("Network weights not saved, file exists.")


def print_parameters(network):
    """ Print the network weights and biases. """
    print("Network weights, bias:")
    for p in network.parameters():
        print(p)


def print_activations(network):
    """ Print the network activations. """
    print("Activations:")
    for child in network.children():
        if isinstance(child, torch.nn.Linear):
            pass
        else:
            print(child.forward())


def save(stuff, name, append_datetime=False):
    """ Save stuff to a file. """
    filename = name
    if append_datetime:
        date_and_time = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename += "-" + date_and_time
    filename += ".pkl"
    with open(filename, "w") as filehandle:
        pickle.dump(stuff, filehandle)


def load(filename):
    """ Load stuff from a file. """
    with open(filename, "r") as filehandle:
        stuff = pickle.load(filehandle)
    return stuff


