# I still have lots to learn about different artificial neural network
# architectures and how to train them, but in order to keep the boss happy
# I'm going to attempt to learn these these things by training them on a very
# simple version of the firefly task.

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
import tensorflow as tf
from ann import Network
import pickle

# define some constants
pi = np.pi


def atan2(y, x):
    """ Tensorflow does not have atan2 yet. Copied this from a comment on
    tensorflow's github page. """
    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0), tf.greater_equal(y,0.0)),
                      tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0), tf.less(y,0.0)),
                      tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)),
                      0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)),
                      -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)),
                      np.nan * tf.zeros_like(x), angle)
    return angle


def frobenius_norm(tensor):
    """ I'm a gonna use this for regularization. """
    square_tensor = tf.square(tensor)
    square_tensor_sum = tf.reduce_sum(square_tensor)
    frobenius_norm = tf.sqrt(square_tensor_sum)
    return frobenius_norm


def tfsub(a, b):
    ax = a[:,0]
    ay = a[:,1]
    bx = b[:,0]
    by = b[:,1]
    return tf.stack([ax - bx, ay - by], 1)


def rect(z):
    """ Convert polar coordinates to rectangular coordinates. """
    if isinstance(z, tf.Tensor):
        angle = z[:,0]
        radius = z[:,1]
        coordinates = tf.stack([radius*tf.cos(angle), radius*tf.sin(angle)], 1)
    elif isinstance(z, np.ndarray):
        angle = z[0][0]
        radius = z[0][1]
        coordinates = np.array([[radius*np.cos(angle), radius*np.sin(angle)]])
    else:
        raise TypeError, "unknown type in rect()"
    return coordinates


def polar(z):
    """ Convert rectangular coordinates to polar coordinates. """
    if isinstance(z, tf.Tensor):
        x = z[:,0]
        y = z[:,1]
        radius = tf.sqrt(tf.square(x) + tf.square(y))
        angle = atan2(y, x)
        #angle = tf.atan(y/x)
        coordinates = tf.stack([radius, angle], 1)
    elif isinstance(z, np.ndarray):
        x = z[0][0]
        y = z[0][1]
        radius = np.sqrt(np.square(x) + np.square(y))
        angle = np.arctan2(y, x)
        coordinates = np.array([[radius, angle]])
    else:
        raise TypeError, "unknown type in polar()"
    return coordinates


def roygbiv(x, n):
    """ Return the RGB value for a color on the electromagnetic spectrum. The
    number of colors to select from the spectrum is specified by n. Which of
    these colors' RGB values should be returned is specified by x (x should
    be between 0 and n-1 inclusive. """
    assert x < n, "x should not be greater than n-1"
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
    neural network. The inputs to the network are the direction and distance to
    the firefly in egocentric coordinates. In this coordinate system the
    direction straight in front of the agent is zero. Angles to the left are
    positive and angles to the right are negative. The outputs of the network
    correspond to a discrete rotation and movement forward or backward of the
    agent. Soft sign activation functions are used to limit how far the agent
    can rotate or move in a single time step. """
    def __init__(self, tf_session, initial_weight_file=None):
        """ The tf_session argument is a tensorflow session. """
        self.tf_session = tf_session
        #self.network_dimensions = [4, 4, 2]
        #self.activation_functions = [tf.nn.relu, tf.tanh]
        self.network_dimensions = [2, 2]
        self.activation_functions = [tf.identity]
        #self.activation_functions = [tf.tanh]
        #self.activation_functions = [tf.nn.softsign]
        self.network = Network(self.network_dimensions,
                               self.activation_functions,
                               initial_weight_file, uniform=False)
        # Two inputs to the network are the distance and direction to the
        # firefly.
        self.direction = self.network.inputs[:,0]
        self.distance = self.network.inputs[:,1] + 0.5
        self.tolerance = 1e-2 # how close the agent has to get
        # The network outputs are the rotation and forward movement of the
        # agent.
        self.rotation = 1.0*pi/2*self.network.outputs[:,0]
        self.step_size = 1.0*self.network.outputs[:,1] + 0.5
        # Using network outputs to update direction and distance.
        self.theta = self.direction - self.rotation
        self.x = self.distance*tf.cos(self.theta)
        self.y = self.distance*tf.sin(self.theta)
        #self.new_direction = tf.atan(self.y/(self.x - self.step_size))
        self.new_direction = atan2(self.y, (self.x - self.step_size))
        self.new_distance2 = (tf.square(self.y)
                              + tf.square(self.x - self.step_size))
        self.objective = (self.new_distance2 +
                          0.0*frobenius_norm(self.network.weights[0]))
        self.new_distance = tf.sqrt(self.new_distance2)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        #self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=10.0)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
                                    #feed_dict=self.feed_dict(2.0/(trial + 1)))
        self.minimize = self.optimizer.minimize(self.objective)
        self.tf_session.run(tf.global_variables_initializer())
        self.training_figure = None
        ### begin debug code
        # set weights to their optimal values
        #weights = np.array([[2/pi, 0.0],
        #                    [0.0, 1.0]], dtype=np.float32)
        #set_weights = self.network.weights[0].assign(weights)
        #self.tf_session.run(set_weights)
        ### end debug code


    def new_trial(self, distance=1.0, verbose=False):
        """ Create a new firefly in front of the agent and no farther away than
        the given distance. """
        firefly = np.array([pi/2*(2*rand(1)[0] - 1), distance*rand(1)[0]])
        if verbose:
            print "Firefly postion (direction, distance):", firefly
        return firefly


    def calc_distance(self, fireflies):
        return np.max(fireflies[:,1])


    def caught(self, fireflies):
        """ Return a True if all fireflies have been caught and False
        otherwise. """
        return self.calc_distance(fireflies) <= self.tolerance


    def feed_dict(self, fireflies):
        """ Generate the feed dictionary from dictionaries that describe the
        location of the agent and the firefly. """
        network_inputs = np.array(fireflies)
        # For fast convergence during training the mean value of each input
        # should be close to zero. The distance input is always positive and
        # during training it is between 0 and 1. Subtract 0.5 so the network
        # sees inputs between -0.5 and 0.5. The 0.5 will be added back for
        # calculation of the new direction and distance.
        network_inputs[:,1] -= 0.5 
        #network_inputs[:,-1] = 1 # constant input
        return {self.network.inputs:network_inputs}


    def eval(self, x, fireflies):
        """ Return the value of a variable or tensor. """
        return self.tf_session.run(x, feed_dict=self.feed_dict(fireflies))


    def show_weight_diffs(self, new_weights, old_weights):
        """ Print the diffences between two sets of network weights. """
        for i in range(len(self.network.weights)):
            print new_weights[i] - old_weights[i]


    def print_weights(self):
        """ Print the network weights. """
        print "Weights:"
        for w in self.network.weights:
            print self.eval(w)


    def print_activations(self):
        print "Activations:"
        for a in self.network.activations:
            print self.eval(a)


    def move(self, fireflies):
        """ Use the network outputs to move the agent and update the firefly's
        position in the agent's egocentric coordinates. Return rotation and
        step_size so the movement can be plotted in a reference frame where the
        firefly is stationary. """
        for i in range(len(fireflies)):
            values = self.eval([self.rotation, self.step_size,
                                self.new_direction, self.new_distance],
                               np.array([fireflies[i]]))
            rotation, step_size, direction, distance = values
            fireflies[i] = np.stack([direction, distance], 1)
        return rotation[0], step_size[0]
        

    def update_progress(self, trial, weights, distances, plot):
        """ Report training progress. """
        print "Practice trial:", trial + 1
        if plot:
            if not self.training_figure:
                self.training_figure = plt.figure()
                axes1 = plt.subplot(211)
                axes2 = plt.subplot(212)
                plt.show(block=False)
            else:
                axes1, axes2 = self.training_figure.get_axes()
                axes1.clear()
                axes1.set_title("Training")
                axes1.set_ylabel("Weights")
                axes2.clear()
                axes2.set_ylabel("Distance")
            for i in range(weights.shape[0]):
                axes1.plot(weights[i,:trial])
            axes2.plot(distances[:trial])
            self.training_figure.canvas.draw()
            self.training_figure.canvas.flush_events()


    def practice(self, batch_size=1, max_trials=1000, distance=1,
                 firefly_file=None, plot=False):
        """ Adjust the network weights to minimize the distance to the firefly
        after taking a step where the size and direction of the step are
        determined by the network outputs. """
        if firefly_file:
            # load fireflies from previous training
            with open(firefly_file, 'r') as file_handle:
                fireflies = pickle.load(file_handle)
        else:
            # generate new fireflies for training
            fireflies = []
            for trial in range(batch_size*max_trials):
                fireflies.append(self.new_trial(distance=distance))
        d = self.network.dimensions
        num_weights = sum([d[i]*d[i+1] for i in range(len(d)-1)])
        weights = np.zeros([num_weights, max_trials])
        distances = np.zeros(max_trials)
        fireflies_caught = 0
        trial = 0
        full = False
        while trial < max_trials and not full:
            batch = np.stack(fireflies[trial*batch_size:(trial+1)*batch_size])
            # train the network
            step = 0
            while step < distance and not self.caught(batch):
                step = step + 1
                self.tf_session.run(self.minimize,
                                    feed_dict=self.feed_dict(batch))
                self.move(batch)
            # count number of fireflies caught in a row
            if self.caught(batch):
                fireflies_caught += batch_size
            else:
                fireflies_caught = 0
            # save weights and distance for plotting
            weights[:,trial] = self.eval(self.network.weights[0],
                                         batch).flatten()
            distances[trial] = self.calc_distance(batch)
            if (trial + 1) % 100 == 0:
                self.update_progress(trial, weights, distances, plot=plot)
                # stop when the performance is good enough
                if fireflies_caught >= 100:
                    full = True
            # increment trial number
            trial = trial + 1
        # print the trained network's weights
        new_weights = self.eval(self.network.weights, batch)
        print new_weights
        if firefly_file == None:
            with open('fireflies.pkl', 'w') as file_handle:
                pickle.dump(fireflies, file_handle)


    def generate_trajectories(self, n, distance=10):
        """ Generate n trajectories using the current network weights. """
        origin = np.zeros(2)
        self.fireflies = []
        self.trajectories = []
        for i in range(n):
            if (i + 1) % 1000 == 0:
                print "Trajectory:", i + 1
            firefly = np.array([self.new_trial(distance=distance)])
            print "firefly location (direction, distance):", firefly
            # Rotate frame of reference 90 degrees for plotting so straight in
            # front of the agent is up instead of to the right.
            self.fireflies.append(np.array(firefly))
            self.fireflies[-1][0][0] += pi/2
            self.fireflies[-1] = rect(self.fireflies[-1])
            #self.print_weights()
            trajectory = [origin]
            agent_direction = pi/2
            steps = 0
            while steps < 3*distance and not self.caught(firefly):
                steps = steps + 1
                rotation, step_size = self.move(firefly)
                agent_direction += rotation
                step = np.array([step_size*np.cos(agent_direction),
                                 step_size*np.sin(agent_direction)])
                trajectory.append(trajectory[-1] + step)
                #print "z:", self.eval(self.network.z[-1])
                #self.print_activations()
                #print "Rotation:", rotation
                #print "Agent direction:", agent_direction
                #print "Step size:", step_size
                #print "Step:", step
                #print "New agent location:", trajectory[-1]
            self.trajectories.append(np.array(trajectory))
            if self.caught(firefly):
                print "Mmmmm, yummy!"
            else:
                print "Final distance to firefly:", self.calc_distance(firefly)


    def plot_trajectories(self):
        """ Plot a trajectory. """
        n = len(self.trajectories)
        if n == 1:
            colors = ["#%06x" % roygbiv(0, 256)]
        else:
            colors = ["#%06x" % roygbiv(i, n) for i in range(n)]
        figure = plt.figure()
        axes = plt.subplot(111)
        axes.set_aspect(1)
        for i, trajectory in enumerate(self.trajectories):
            x = trajectory[:,0]
            y = trajectory[:,1]
            axes.plot(x, y, color=colors[i])
            firefly_x = self.fireflies[i][0][0]
            firefly_y = self.fireflies[i][0][1]
            axes.plot(firefly_x, firefly_y, color=colors[i], marker='o')
        plt.show(block=False)


    def plot_objective_function(self):
        #w0 = self.eval(self.network.weights[0])
        #w0 = np.array([[2/pi, 0],[0, 1.0]])
        #w0 = np.array([[2/pi, 1.3],[1.6, 1.0]])
        w0 = np.array([[2.0, 0.6],[3.0, 1.0]])
        figure = plt.figure()
        axes = plt.subplot(111)
        self.firefly = np.array([[pi/4, 1.0]])
        dw = 0.1
        weight_range = np.arange(-4, 4+dw, dw)
        f = np.zeros(len(weight_range))
        new_weights = tf.placeholder(tf.float32, shape=w0.shape)
        set_weights = self.network.weights[0].assign(new_weights)
        for col in range(w0.shape[1]):
            for row in range(w0.shape[0]):
                weights = np.array(w0)
                for i, w in enumerate(weight_range):
                    weights[row,col] = w
                    self.tf_session.run(set_weights,
                                        feed_dict={new_weights:weights})
                    f[i] = self.eval(self.objective)
                axes.plot(weight_range, f, label="%d, %d" % (row,col))
        handles, labels = axes.get_legend_handles_labels()
        axes.set_xlabel("Weight")
        axes.set_ylabel("Objective Function Value")
        axes.legend(handles, labels, ncol=2)
        plt.show(block=False)
        # reset weights to w0
        self.tf_session.run(set_weights, feed_dict={new_weights:w0})


def plot_training_histogram(n, max_trials, batch_size=1):
    """ Train the network n times and plot histograms of each weight in the
    network. This gives a good idea of how robust the training method is. """
    game = FireflyTask(session)
    d = game.network.dimensions
    num_weights = sum([d[i]*d[i+1] for i in range(len(d)-1)])
    weights = np.zeros([num_weights, n])
    for i in range(n):
        game.practice(batch_size=batch_size, max_trials=max_trials, distance=1)
        weights[:,i] = game.tf_session.run(game.network.weights[0]).flatten()
        game = FireflyTask(session)
    fig1 = plt.figure()
    for i in range(num_weights):
        axes = plt.subplot("22%d" % (i+1))
        axes.hist(weights[i], 50)
    axes.set_xlabel("weight value")
    plt.show(block=False)


if __name__ == "__main__":
    print "Let's eat some fireflies, yeah!"
    initial_weights = None
    fireflies = None
    #initial_weights = "initial_weights1242.pkl"
    #fireflies = "fireflies1242.pkl"
    session = tf.Session()
    game = FireflyTask(session, initial_weight_file=initial_weights)
    #game.plot_objective_function()
    #game.print_weights()
    #game.practice(batch_size=1, max_trials=5000, distance=1,
    #              firefly_file=fireflies, plot=True)
    #game.generate_trajectories(10, distance=10)
    #game.plot_trajectories()
    plot_training_histogram(100, 1000, 4)
