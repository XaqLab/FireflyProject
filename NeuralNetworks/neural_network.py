# I want to train a neural network to learn a max function and I'm going to
# start with the very simple case of a two input network.

import numpy as np
from numpy.random import rand, randn
import tensorflow as tf
import pickle
import os

DEBUG = True


class NeuralNetwork(object):
    """ A simple artificial neural network class. """
    def __init__(self, tf_session, dimensions, activation_functions,
                 uniform=True):
        """ The dimensions argument is a list that specifies the number of
        inputs to the network and the number of units in each layer.  The
        number of inputs to the network is dimensions[0] and the number of
        units in the nth layer is dimensions[n].  The activation_functions
        argument is a list of functions to be used as the activation fuction
        for each layer of the network. """
        self.tf_session = tf_session
        self.dimensions = dimensions
        self.inputs = tf.placeholder(tf.float32,
                                     shape=[None, dimensions[0]])
        self.weights = []
        self.z = []
        self.activations = [self.inputs]
        # generate new initial weights
        initial_weights = []
        for i in range(len(dimensions) - 1):
            fan_in = dimensions[i]
            fan_out = dimensions[i+1]
            if uniform:
                # Xavier initialization, uniform distribution
                x = np.sqrt(6.0 / (fan_in + fan_out))
                iw = 2*x*rand(dimensions[i], dimensions[i+1]) - x
            else:
                # Xavier initialization, normal distribution
                sigma = np.sqrt(3.0 / (fan_in + fan_out))
                iw = sigma*randn(dimensions[i], dimensions[i+1])
            iw = np.array(iw, dtype=np.float32)
            initial_weights.append(iw)
        for i in range(len(dimensions) - 1):
            W = tf.Variable(initial_weights[i])
            self.weights.append(W)
            self.z.append(tf.matmul(self.activations[-1], W))
            f = activation_functions[i]
            # Use different activation functions within a layer
            #alist = [f[j](z[:,j]) for j in range(dimensions[i+1])]
            #activations = tf.concat(alist, 1)
            #activations = tf.pack(alist, 0)
            activations = f(self.z[-1])
            self.activations.append(activations)
	self.outputs = self.activations[-1]


    def set_weights(self, weights):
        """ If the argument is a list then set the network weights using the
        weights in the list. If the argument is a filename then read the
        weights from the file. """
        if isinstance(weights, str):
            weight_file = weights
            with open(weight_file, 'r') as file_handle:
                weight_list = pickle.load(file_handle)
        if isinstance(weights, list):
                weight_list = weights
        for i in range(len(weight_list)):
            self.tf_session.run(self.weights[i].assign(weight_list[i]))


    def save_weights(self, filename):
        """ Save the network's weights. """
        if not os.path.isfile(filename):
            # Don't overwrite existing files.
            weights = self.tf_session.run(self.weights)
            with open(filename, 'w') as file_handle:
                pickle.dump(self.tf_session.run(self.weights), file_handle)
        else:
            # Let the user know the weights were not saved.
            raise IOError, "Network weights not saved, file exists."


    def print_weights(self):
        """ Print the network weights. """
        print "Network weights:"
        for w in self.weights:
            print self.tf_session.run(w)


    def print_activations(self, inputs):
        """ Print the network activations. """
        print "Activations:"
        for a in self.network.activations:
            print self.tf_session.run(w, feed_dict={self.inputs: inputs})


    def infer(self, inputs):
        """ Given the inputs return the network outputs. """
        return self.tf_session.run(self.activations[-1],
                                   feed_dict={self.inputs:inputs})


    def learn(self, inputs, outputs):
        """ Given the desired outputs for a set of inputs, use the learning
        rule to update the weights until the outputs of the network match the
        desired outputs. """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
	network_outputs = self.activations[-1]
        desired_outputs = tf.placeholder(tf.float32,
                                         shape=network_outputs.get_shape())
        error = tf.reduce_sum(tf.squared_difference(desired_outputs,
                                                    network_outputs))
        minimize = optimizer.minimize(error)
        training_data = {self.inputs:inputs.astype(np.float32),
                         desired_outputs:outputs.astype(np.float32)}
        previous_training_error = 1e12
        training_error = 0
        iteration = 0
        while abs(previous_training_error - training_error) > 1e-6:
            iteration = iteration + 1
            #for i in range(len(self.weights)):
            #    a = self.activations[i]
            #    print "activations:"
            #    print a.eval(session=self.tf_session, feed_dict=training_data)
            #    W = self.weights[i]
            #    print "weights:"
            #    print W.eval(session=self.tf_session)
            #a = self.activations[-1]
            #print "activations:"
            #print a.eval(session=self.tf_session, feed_dict=training_data)
            if iteration % 100 == 0:
                print "Training error:", 
                print training_error
            training_error = self.tf_session.run(error,feed_dict=training_data)
            #print training_error
            previous_training_error = training_error
            self.tf_session.run(minimize, feed_dict=training_data)
            training_error = self.tf_session.run(error,feed_dict=training_data)
            #print training_error
        print "%.6f" % previous_training_error
        print "%.6f" % training_error


