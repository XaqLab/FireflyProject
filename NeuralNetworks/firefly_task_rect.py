""" Only import packages needed for training the network. Network training will
sometimes be done on a remote server so no matplotlib, etc. """
import numpy as np
from numpy.random import rand, randn
import tensorflow as tf
import pickle
import signal
from neural_network import NeuralNetwork
from math import isnan


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


# define some constants
pi = np.pi


def atan2(y, x, last_angle):
    """ Tensorflow does not have atan2 yet. Copied this from a comment on
    tensorflow's github page. """
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0), tf.greater_equal(y,0.0)),
                      tf.atan(y/x) + pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0), tf.less(y,0.0)),
                      tf.atan(y/x) - pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)),
                      0.5*pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)),
                      -0.5*pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)),
                      np.nan * tf.zeros_like(x), angle)
    # The value of atan2(y,x) is ambigous. Adding 2*pi to any solution produces
    # another solution. This results in a discontinuity in atan2. When x is
    # negative and y goes from a small positive number to a small negative
    # number, the output of the function goes from pi to -pi. This
    # discontinuity is bad for optimization methods like SGD, Adam, and
    # Adagrad. To eliminate this discontinuity return the value in the list
    # [angle - 2*pi, angle, angle + 2*pi] that is closest to the previous
    # angle.
    delta_angle_minus_2pi = tf.abs(angle - 2*pi - last_angle)
    delta_angle = tf.abs(angle - last_angle)
    delta_angle_plus_2pi = tf.abs(angle + 2*pi - last_angle)
    angle = tf.where(tf.less(delta_angle_minus_2pi, delta_angle),
                     angle - 2*pi, angle)
    angle = tf.where(tf.less(delta_angle_plus_2pi, delta_angle),
                     angle + 2*pi, angle)
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
        #angle = atan2(y, x)
        angle = tf.atan(y/x)
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
    def __init__(self, tf_session, network):
        """ The tf_session argument is a tensorflow session. """
        self.tf_session = tf_session
        if network:
            self.network_dimensions = network['dimensions']
            self.activation_functions = network['activation functions']
            self.optimizer = network['optimizer']
        else:
            self.network_dimensions = [3, 100, 10, 2]
            scaled_tanh = lambda x: 1.7159*tf.tanh(2.0*x/3.0)
            self.activation_functions = [scaled_tanh]*3
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
            #self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=10.0)
            #self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        #self.network_dimensions = [3,  2]
        #self.activation_functions = [tf.identity]*2
        #self.activation_functions = [tf.tanh]*3
        #self.activation_functions = [tf.nn.softsign]
        #self.activation_functions = [tf.nn.relu, tf.nn.relu, tf.tanh]
        #self.activation_functions = [tf.nn.softsign, tf.nn.softsign]
        self.afun_names = [f.__name__ for f in self.activation_functions]
        self.network = NeuralNetwork(self.tf_session, self.network_dimensions,
                                     self.activation_functions, uniform=False)
        # Two inputs to the network are the distance and direction to the
        # firefly.
        self.x = self.network.inputs[:,0]
        self.y = self.network.inputs[:,1]
        self.distance = tf.sqrt(tf.square(self.x) + tf.square(self.y))
        self.tolerance = 1e-2 # how close the agent has to get
        #self.tolerance = 2e-1 # how close the agent has to get
        # The network outputs are the rotation and forward movement of the
        # agent.
        self.dx = self.network.outputs[:,0]
        self.dy = self.network.outputs[:,1]
        # Normalize dx and dy so the step size <= 1
        self.step_size = tf.sqrt(tf.square(self.dx) + tf.square(self.dy))
        self.dx = tf.where(tf.greater(self.step_size,1.0),
                           self.dx/self.step_size, self.dx)
        self.dy = tf.where(tf.greater(self.step_size,1.0),
                           self.dy/self.step_size, self.dy)
        self.new_x = self.x - self.dx
        self.new_y = self.y - self.dy
        self.new_distance2 = tf.square(self.new_x) + tf.square(self.new_y)
        self.new_distance = tf.sqrt(self.new_distance2)
        #self.objective = (self.new_distance2 +
        #                  0.0*frobenius_norm(self.network.weights[0]))
        self.objective = tf.nn.softsign(self.new_distance - self.distance)
        #self.objective = tf.nn.softsign(self.new_distance/self.distance - 1)
        self.minimize = self.optimizer.minimize(self.objective)
        self.tf_session.run(tf.global_variables_initializer())
        self.training_figure = None


    def new_trial(self, distance=1.0, verbose=False):
        """ Create a new firefly in front of the agent and no farther away than
        the given distance. """
        distance_to_firefly = distance*rand(1)[0]
        direction_to_firefly = pi/2*(2*rand(1)[0] - 1)
        x = distance_to_firefly*np.sin(direction_to_firefly)
        y = distance_to_firefly*np.cos(direction_to_firefly)
        firefly = np.array([x,y])
        if verbose:
            print "Firefly postion (x, y):", firefly
        return firefly


    def calc_distance(self, fireflies):
        distances = np.sqrt([f[0]**2 + f[1]**2 for f in fireflies])
        return np.max(distances)


    def caught(self, fireflies):
        """ Return a True if all fireflies have been caught and False
        otherwise. """
        return self.calc_distance(fireflies) <= self.tolerance


    def feed_dict(self, fireflies):
        """ Generate the feed dictionary from dictionaries that describe the
        location of the agent and the firefly. """
        rows, cols = fireflies.shape
        network_inputs = np.zeros([rows, self.network.dimensions[0]])
        network_inputs[:,0:cols] = fireflies
        # For fast convergence during training the mean value of each input
        # should be close to zero. The distance input is always positive and
        # during training it is between 0 and 1. Subtract 0.5 so the network
        # sees inputs between -0.5 and 0.5. The 0.5 will be added back for
        # calculation of the new direction and distance.
        #network_inputs[:,1] -= 0.5 
        network_inputs[:,-1] = 1 # constant input
        return {self.network.inputs:network_inputs}


    def eval(self, x, fireflies):
        """ Return the value of a variable or tensor. """
        return self.tf_session.run(x, feed_dict=self.feed_dict(fireflies))


    def show_weight_diffs(self, new_weights, old_weights):
        """ Print the diffences between two sets of network weights. """
        for i in range(len(self.network.weights)):
            print new_weights[i] - old_weights[i]


    def move(self, fireflies):
        """ Use the network outputs to move the agent and update the firefly's
        position in the agent's egocentric coordinates. Return rotation and
        step_size so the movement can be plotted in a reference frame where the
        firefly is stationary. """
        for i in range(len(fireflies)):
            values = self.eval([self.dx, self.dy, self.new_x, self.new_y],
                               np.array([fireflies[i]]))
            dx, dy, x, y = values
            fireflies[i] = np.stack([x, y], 1)
        return dx[0], dy[0]
        

    def practice(self, batch_size=1, max_trials=1000, distance=1,
                 firefly_file=None, plot_progress=False):
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
        layers = len(self.network.dimensions) - 1
        num_weights = sum([d[i]*d[i+1] for i in range(len(d)-1)])
        weights = None
        dw_means = np.zeros([layers, max_trials - 1])
        dw_stds = np.zeros([layers, max_trials - 1])
        distances = np.zeros(max_trials)
        fireflies_caught = 0
        trial = 0
        full = False
        with GracefulInterruptHandler() as h:
            while trial < max_trials and not full:
                batch = np.stack(fireflies[trial*batch_size:(trial+1)*batch_size])
                # train the network
                step = 0
                while step < 3*distance and not self.caught(batch):
                    step = step + 1
                    W0 = self.eval(self.network.weights, batch)
                    self.tf_session.run(self.minimize,
                                        feed_dict=self.feed_dict(batch))
                    W = self.eval(self.network.weights, batch)
                    if any([isnan(a) for c in W for b in c for a in b]):
                        print "Yo, your weights are NaN bro."
                        import ipdb;ipdb.set_trace()
                    self.move(batch)
                # count number of fireflies caught in a row
                if self.caught(batch):
                    fireflies_caught += batch_size
                else:
                    fireflies_caught = 0
                # save distance and mean and std of weight updates for plotting
                distances[trial] = self.calc_distance(batch)
                if trial > 0:
                    old_weights = weights
                    weights = [w for w in self.eval(self.network.weights, batch)]
                    dws = [weights[i] - old_weights[i]
                           for i in range(len(weights))]
                    dw_means[:,trial-1] = np.array([dw.mean() for dw in dws])
                    dw_stds[:,trial-1] = np.array([dw.std() for dw in dws])
                else:
                    weights = [w for w in self.eval(self.network.weights, batch)]
                if (trial + 1) % 100 == 0:
                    print "Practice trial:", trial + 1, \
                            "mean distance:", distances[:trial].mean()
                    if plot_progress:
                        plot_progress(self, trial, dw_means, dw_stds,
                                      distances[:trial], batch_size)
                    # stop when the performance is good enough
                    if fireflies_caught >= 100:
                        full = True
                # increment trial number
                trial = trial + 1
                if h.interrupted:
                    # allow the user to stop training using Control-C
                    break
        return trial, fireflies


    def generate_trajectories(self, n, distance=10):
        """ Generate n trajectories using the current network weights. """
        origin = np.zeros(2)
        fireflies = []
        trajectories = []
        final_distances = []
        for i in range(n):
            if (i + 1) % 1000 == 0:
                print "Trajectory:", i + 1
            firefly = np.array([self.new_trial(distance=distance)])
            print "firefly location (x, y):", firefly
            # Rotate frame of reference 90 degrees for plotting so straight in
            # front of the agent is up instead of to the right.
            fireflies.append(np.array(firefly))
            #self.print_weights()
            trajectory = [origin]
            agent_direction = pi/2
            steps = 0
            while steps < 3*distance and not self.caught(firefly):
                steps = steps + 1
                dx, dy = self.move(firefly)
                step = np.array([dx, dy])
                trajectory.append(trajectory[-1] + step)
                #print "z:", self.eval(self.network.z[-1])
                #self.print_activations()
                #print "Rotation:", rotation
                #print "Agent direction:", agent_direction
                #print "Step size:", step_size
                #print "Step:", step
                #print "New agent location:", trajectory[-1]
            trajectories.append(np.array(trajectory))
            final_distance = self.calc_distance(firefly)
            final_distances.append(np.array(final_distance))
            if self.caught(firefly):
                print "Mmmmm, yummy!"
            else:
                print "Final distance to firefly:", final_distance
        return trajectories, fireflies, final_distances


