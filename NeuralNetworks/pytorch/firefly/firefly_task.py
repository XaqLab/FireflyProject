""" Only import packages needed for training the network. Network training will
sometimes be done on a remote server so no matplotlib, etc. """
import numpy as np
from numpy.random import rand, randn
import torch
from torch.autograd import Variable
import pickle
import signal
from neural_network import to_array
from keypress import Keypress

softsign = torch.nn.Softsign()

# define some constants
pi = np.pi


def rect(z):
    """ Convert polar coordinates to rectangular coordinates. """
    if isinstance(z, np.ndarray):
        angle = z[0][0]
        radius = z[0][1]
        coordinates = np.array([[radius*np.cos(angle), radius*np.sin(angle)]])
    else:
        raise TypeError("unknown type in rect()")
    return coordinates


def polar(z):
    """ Convert rectangular coordinates to polar coordinates. """
    if isinstance(z, np.ndarray):
        x = z[0][0]
        y = z[0][1]
        radius = np.sqrt(np.square(x) + np.square(y))
        angle = np.arctan2(y, x)
        coordinates = np.array([[radius, angle]])
    else:
        raise TypeError("unknown type in polar()")
    return coordinates


def new_trial(max_distance=1.0, verbose=False):
    """ Create a new firefly in front of the agent and no farther away than
    the given distance. """
    direction = pi/2*(2*rand(1)[0] - 1)
    distance = max_distance*rand(1)[0]
    firefly = Variable(torch.FloatTensor([[direction, distance]]))
    if verbose:
        print("Firefly postion (direction, distance):", firefly)
    return firefly


def move(firefly, network_output):
    """ Use the network outputs to move the agent in egocentric coordinates.
    Return rotation and step_size so the trajectory can be plotted in a
    reference frame where the firefly is stationary. Return the updated
    direction and distance to the firefly in new_firefly. """
    direction = firefly[0,0]
    distance = firefly[0,1]
    rotation = np.pi/2*network_output[0,0]
    step_size = network_output[0,1]
    theta = direction - rotation
    x = distance*torch.cos(theta)
    y = distance*torch.sin(theta)
    #atan2 = torch.atan2()
    #new_direction = torch.atan2(y.data, (x.data - step_size.data))
    new_direction = torch.atan(y/(x - step_size))
    new_distance2 = (y.pow(2) + (x - step_size).pow(2))
    new_distance = torch.sqrt(new_distance2)
    new_firefly = torch.stack([new_direction, new_distance], dim=1)
    return rotation, step_size, new_firefly
        

def calc_distance(fireflies):
    return torch.max(fireflies[:,1]).data[0]


def caught(fireflies, tolerance=1e-2):
    """ Return a True if all fireflies have been caught and False
    otherwise. """
    return calc_distance(fireflies) <= tolerance


def generate_trajectories(network, n, max_distance=10):
    """ Generate n trajectories using the current network weights. """
    origin = np.zeros(2)
    fireflies = []
    trajectories = []
    final_distances = []
    for i in range(n):
        if (i + 1) % 1000 == 0:
            print("Trajectory:", i + 1)
        firefly = new_trial(max_distance=max_distance)
        print("firefly location (direction, distance):", to_array(firefly))
        # Rotate frame of reference 90 degrees for plotting so straight in
        # front of the agent is up instead of to the right.
        fireflies.append(to_array(firefly))
        fireflies[-1][0][0] += pi/2
        fireflies[-1] = rect(fireflies[-1])
        trajectory = [origin]
        agent_direction = pi/2
        steps = 0
        old_distance = calc_distance(firefly)
        #while not caught(firefly) and steps < 30*max_distance:
        while not caught(firefly) and (steps == 0 or distance_change < 0):
            steps = steps + 1
            rotation, step_size, firefly = move(firefly, network(firefly))
            new_distance = calc_distance(firefly)
            distance_change = new_distance - old_distance
            old_distance = new_distance
            agent_direction += rotation
            dx = step_size*torch.cos(agent_direction)
            dy = step_size*torch.sin(agent_direction)
            step = np.array([dx.data[0], dy.data[0]])
            trajectory.append(trajectory[-1] + step)
        trajectories.append(np.array(trajectory))
        final_distance = calc_distance(firefly)
        final_distances.append(np.array(final_distance))
        if caught(firefly):
            print("Mmmmm, yummy!")
        else:
            print("Final distance to firefly:", final_distance)
    return trajectories, fireflies, final_distances


def practice(self, batch_size=1, max_trials=1000, distance=1,
             fireflies=None, plot_progress=False):
    """ Adjust the network weights to minimize the distance to the firefly
    after taking a step where the size and direction of the step are
    determined by the network outputs. """
    if fireflies == None:
        # generate new fireflies for training
        fireflies = []
        for trial in range(batch_size*max_trials):
            fireflies.append(self.new_trial(distance=distance))
    d = self.network.dimensions
    layers = len(self.network.dimensions) - 1
    weights = None
    dw_means = np.zeros([layers, max_trials - 1])
    dw_stds = np.zeros([layers, max_trials - 1])
    distances = np.zeros(max_trials)
    fireflies_caught = 0
    trial = 0
    full = False
    learning_rate = 0.1
    lr_delta = 10**round(np.log10(learning_rate))
    keypress = Keypress()
    with GracefulInterruptHandler() as h:
        while trial < max_trials and not full:
            batch = np.stack(fireflies[trial*batch_size:(trial+1)*batch_size])
            # train the network
            step = 0
            old_distance = self.calc_distance(batch)
            #while step < distance and not self.caught(batch):
            while (not self.caught(batch)
                   and (step == 0 or distance_change < 0)):
                step = step + 1
                self.optimizer.zero_grad()
                self.objective.backward()
                self.optimizer.step()
                self.move(batch)
                new_distance = self.calc_distance(batch)
                distance_change = new_distance - old_distance
                old_distance = new_distance
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
                print("Practice trial:", trial + 1, \
                        "mean distance:",
                      distances[trial - 99:trial + 1].mean())
                if plot_progress:
                    plot_progress(self, trial, dw_means, dw_stds,
                                  distances[:trial], batch_size)
                # Stop when the performance is good enough.
                if fireflies_caught >= 100:
                    full = True
            # Increment the trial number.
            trial = trial + 1
            if h.interrupted:
                # Allow user to stop training using Control-C.
                break
            if self.optimizer.__dict__['_name'] == 'GradientDescent':
                # For gradient descent allow user to change learning rate
                # interactively.
                key = keypress()
                if key == 'up':
                    learning_rate += lr_delta
                    print("LR =", learning_rate)
                if key == 'down':
                    learning_rate -= lr_delta
                    print("LR =", learning_rate)
                if key == 'left':
                    lr_delta *= 10
                if key == 'right':
                    lr_delta *= 0.1
    return fireflies


