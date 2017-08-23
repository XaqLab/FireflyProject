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


def closure(function, data, *args, **kwargs):
    """ Enclose the function and data so the data is remembered between calls
    to function. """
    enclosed_data = data
    return lambda *args, **kwargs: function(enclosed_data, *args, **kwargs)


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
    direction = pi*rand(1)[0]
    distance = max_distance*rand(1)[0]
    x = distance * np.cos(direction)
    y = distance * np.sin(direction)
    firefly = Variable(torch.FloatTensor([[x, y]]))
    if verbose:
        print("Firefly postion (direction, distance):", firefly)
    return firefly


def encode(network_input):
    """ Shift and scale the network input to get a mean of zero and a variance
    of one.  """
    #offset = Variable(torch.Tensor([[0.0,-0.5]]))
    offset = Variable(torch.Tensor([[0.0, 0.0]]))
    return network_input + offset


def move(agent, firefly, network, hx):
    """ Use the network outputs to move the agent in absolute coordinates.
    reference frame where the firefly is stationary. Return the updated
    agent postion new_agent. """
    #network_output = network(encode(firefly - agent), hx)
    network_output = network(encode(firefly - agent))
    x_step = network_output[0,0]
    y_step = network_output[0,1]
    x = agent[0,0]
    y = agent[0,1]
    new_x = x + x_step
    new_y = y + y_step
    new_agent = torch.stack([new_x, new_y], dim=1)
    return new_agent
        

def calc_distance(firefly, agent):
    return torch.norm(firefly - agent).data[0]


def caught(firefly, agent, tolerance=1e-1):
    """ Return True if the firefly has been caught and False otherwise. """
    return calc_distance(firefly, agent) <= tolerance


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
        agent = Variable(torch.zeros([1,2]))
        #hx = Variable(torch.zeros([1,2]))
        hx = Variable(torch.randn([1,2]))
        print("firefly location (x, y):", to_array(firefly))
        # Rotate frame of reference 90 degrees for plotting so straight in
        # front of the agent is up instead of to the right.
        fireflies.append(to_array(firefly))
        trajectory = [origin]
        steps = 0
        old_distance = calc_distance(firefly, agent)
        #while not caught(firefly) and steps < 30*max_distance:
        while not caught(firefly, agent) and (steps == 0 or distance_change < 0):
            steps = steps + 1
            #hx = network(firefly - agent, hx)
            #agent = move(agent, hx)
            agent = move(agent, firefly, network, hx)
            new_distance = calc_distance(firefly, agent)
            distance_change = new_distance - old_distance
            old_distance = new_distance
            trajectory.append([agent.data[0,0], agent.data[0,1]])
        trajectories.append(np.array(trajectory))
        final_distance = calc_distance(firefly, agent)
        final_distances.append(np.array(final_distance))
        if caught(firefly, agent):
            print("Mmmmm, yummy!")
        else:
            print("Final distance to firefly:", final_distance)
    return trajectories, fireflies, final_distances



"""
I want to divide the functionality of this method into:
    - a class for graphing data during training
    ? something to catch Control-C to stop training and graph trajectories
    ? something to allow interactive adjustment of the learning rate for SGD


"""
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


