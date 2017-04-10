# I still have lots to learn about different artificial neural network
# architectures and how to train them, but in order to keep the boss happy
# I'm going to attempt to learn these these things by training them on a very
# simple version of the firefly task.

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plot
import tensorflow as tf
from ann import Network


def roygbiv(x, n):
    """ Return the RGB value for a color on the electromagnetic spectrum. The
    number of colors to select from the spectrum is specified by n. Which of
    these colors' RGB values should be returned is specified by x (x should
    be between 0 and n-1 inclusive. """
    assert x < n, "x should not be greater than n-1"
    pi = np.pi
    w = 5.0/3.0*pi
    phi = 2.0/3.0*pi
    i = float(x)
    r = int(255*(np.cos(w*i/(n - 1)) + 1)/2.0)
    g = int(255*(np.cos(w*i/(n - 1) - phi) + 1)/2.0)
    b = int(255*(np.cos(w*i/(n - 1) - 2*phi) + 1)/2.0)
    return 2**16*r + 2**8*g + b


class FireflyTask(object):
    """ A simple task that requires the subject/agent to navigate to a target
    that is initially visible and then disappears. I'm going to start with a
    target that is always visible. The agent is represented by an artificial
    neural network. The inputs to the network are the firefly's position and
    the agent's position in Cartesian coordinates. The outputs of the network
    are the agent's forward and lateral velocities.  Rectified linear units
    serve as the non-linear activation function for all layers except the last
    which uses a tanh to limit the agent's velocities. """
    def __init__(self, tf_session):
        """ The tf_session argument is a tensorflow session and the
        network_dimensions argument is a list that specifies the number of
        units in each layer of the network. """
        self.tf_session = tf_session
        #self.network_dimensions = [4, 4, 2]
        #self.activation_functions = [tf.nn.relu, tf.tanh]
        self.network_dimensions = [4, 2]
        #self.activation_functions = [tf.tanh]
        self.activation_functions = [tf.nn.softsign]
        self.network = Network(self.network_dimensions,
                               self.activation_functions)
        self.firefly_position = self.network.inputs[:,0:2]
        self.agent_position = self.network.inputs[:,2:4]
        self.tolerance = 1e-2 # how close the agent has to get
        self.agent_move = self.network.outputs
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        #self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        new_agent_position = self.agent_position + self.agent_move
        diff2 = tf.squared_difference(self.firefly_position,new_agent_position)
        distance2 = tf.reduce_sum(diff2)
        self.distance = tf.sqrt(distance2)
        self.minimize = self.optimizer.minimize(distance2)
        self.tf_session.run(tf.global_variables_initializer())
        self.firefly = 10*rand(2)
        self.agent = np.zeros([1,2])
        ### begin debug code
        # set weights to their optimal values
        #weights = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        #                   dtype=np.float32)
        #set_weights = self.network.weights[0].assign(weights)
        #self.tf_session.run(set_weights)
        ### end debug code


    def new_trial(self, distance=1, verbose=False):
        """ Move the agent back to the origin and the firefly to a random
        location. """
        self.firefly = distance*randn(2)
        self.agent = np.zeros([1,2])
        if verbose:
            print "Firefly postion:", self.firefly
            print "Agent position:", self.agent,
            print "Distance to firefly:", self.calc_distance()


    def calc_distance(self):
        return np.sqrt(((self.firefly - self.agent)**2).sum())


    def caught_firefly(self):
        return game.calc_distance() <= self.tolerance


    def feed_dict(self):
        """ Generate the feed dictionary from dictionaries that describe the
        location of the agent and the firefly. """
        network_inputs = np.zeros([1, self.network_dimensions[0]])
        network_inputs[:,0:2] = self.firefly
        network_inputs[:,2:4] = self.agent
        return {self.network.inputs:network_inputs}


    def eval(self, x):
        """ Return the value of a variable or tensor. """
        return self.tf_session.run(x, feed_dict=self.feed_dict())


    def show_weight_diffs(self, new_weights, old_weights):
        """ Print the diffences between two sets of network weights. """
        for i in range(len(self.network.weights)):
            print new_weights[i] - old_weights[i]


    def print_activations():
        print "Activations:"
        for a in self.network.activations:
            print self.eval(a)


    def move(self, verbose=False):
        """ Update the agent's position. """
        move = self.tf_session.run(self.agent_move, feed_dict=self.feed_dict())
        self.agent = self.agent + move
        if verbose:
            print "Agent position:", self.agent,
            print "Distance to firefly:", self.calc_distance()
        

    def practice(self, trials=1):
        """ Adjust the weights to minimize the difference between the agent's
        location and the firefly's location. """
        old_weights = self.eval(self.network.weights)
        for trial in range(trials):
            self.new_trial()
            if (trial + 1) % 100 == 0:
                print "Practice trial:", trial + 1
            step = 0
            while step < 5 and not self.caught_firefly():
                step = step + 1
                self.tf_session.run(self.minimize, feed_dict=self.feed_dict())
                game.move()
        new_weights = self.eval(self.network.weights)
        print new_weights
        #self.show_weight_diffs(new_weights, old_weights)
        #print self.eval(self.network.z)
        #print self.eval(self.network.activations)


    def generate_trajectories(self, n):
        """ Generate n trajectories using the current network weights. """
        self.fireflies = []
        self.trajectories = []
        for i in range(n):
            #self.new_trial(distance=10, verbose=True)
            self.new_trial(distance=10)
            trajectory = [self.agent]
            step = 0
            while step < 50 and not self.caught_firefly():
                step = step + 1
                #game.move(verbose=True)
                game.move()
                trajectory.append(self.agent)
            self.fireflies.append(self.firefly)
            self.trajectories.append(np.array(trajectory))


    def plot_trajectories(self):
        """ Plot a trajectory. """
        n = len(self.trajectories)
        if n == 1:
            colors = ["#%06x" % roygbiv(0, 256)]
        else:
            colors = ["#%06x" % roygbiv(i, n) for i in range(n)]
        figure = plot.figure()
        axes = plot.subplot(111)
        axes.set_aspect(1)
        for i, trajectory in enumerate(self.trajectories):
            x = trajectory[:,0,0]
            y = trajectory[:,0,1]
            axes.plot(x, y, color=colors[i])
            firefly_x = self.fireflies[i][0]
            firefly_y = self.fireflies[i][1]
            axes.plot(firefly_x, firefly_y, color=colors[i], marker='o')
        plot.show(block=False)


if __name__ == "__main__":
    session = tf.Session()
    print "Let's eat some fireflies, yeah!"
    game = FireflyTask(session)
    game.practice(trials=1000)
    game.generate_trajectories(10)
    game.plot_trajectories()
    if game.caught_firefly():
        print "Mmmmm, yummy!"
