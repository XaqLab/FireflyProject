""" Only import packages needed for training the network. Network training will
sometimes be done on a remote server so no matplotlib, etc. """
import numpy as np
from numpy.random import rand, randn
import torch
import pickle
import signal
import datetime as dt
from neural_network import *
from firefly_task import *
#from firefly_task_rect import FireflyTask
from visualization import load


# default to no plots for batch training on server
plot_progress = False
if "__IPYTHON__" in __builtins__.keys():
    # plot progress for interactive training in iPython
    from visualization import *


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


def train_repeatedly(network, n, max_trials, batch_size=1):
    """ Train the network n times and save the weights. Each weight can then be
    plotted as a histogram to see how robust the training method is. """
    game = FireflyTask()
    weights = []
    num_trials = []
    for i in range(n):
        trials, _ = game.practice(batch_size=batch_size, max_trials=max_trials,
                                  distance=1)
        weights.append(game.network.get_weights)
        num_trials.append(trials)
        #game = FireflyTask(tf_session)
    return weights, num_trials


def generate_trajectories2(networks, num_trajectories, distance):
    """ Generate num_trajectories for each network in networks. """
    game = FireflyTask(session)
    fireflies = []
    trajectories = []
    final_distances = []
    for network in networks:
        game = FireflyTask(session)
        game.network.set_weights(network)
        returns = game.generate_trajectories(num_trajectories, distance)
        trajectory, firefly, final_distance = returns
        trajectories += trajectory
        fireflies += firefly
        final_distances += final_distance
    return trajectories, fireflies, final_distances


if __name__ == "__main__":
    print("Let's eat some fireflies, yeah!")
    #### create network ####
    dimensions = [2, 2]
    #scaled_tanh = lambda x: 1.7159*torch.tanh(2.0*x/3.0)
    #activation_functions = [scaled_tanh]*2
    #activation_functions = [torch.nn.ReLU()]
    activation_functions = [torch.nn.Tanh()]
    network = create_network(dimensions, activation_functions)
    #### initialize network ####
    #initial_weights = session.run(game.network.weights)
    #save([initial_weights], "initial_weights", append_datetime=True)
    #weights = [np.array([[ 0.63661977,  0.00000000],
    weights = [np.array([[ 1.0,  0.0],
                         [ 0.0,  1.0]], dtype=np.float32)]
    biases = [np.array([ 0.0,  0.0], dtype=np.float32)]
    hand_picked = {'weights':weights, 'biases':biases}
    set_parameters(network, hand_picked)
    #### train network ####
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    print_parameters(network)

    # enable interruption of training and still continue code execution
    for i in range(1000):
        firefly = new_trial()
        _, _, firefly = move(firefly, network(firefly))
        direction = firefly[0,0]
        distance = firefly[0,1]
        cost = distance.pow(2)
        print(i, cost.data[0])
        # plot cost while training
        # plot activations, 1 plot w/1 curve for each layer
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # add option to save all training data
        # add option to manually adjust learning rate while training

    print_parameters(network)

    #fireflies = game.practice(batch_size=1, max_trials=20000, distance=1,
    #                          fireflies=None, plot_progress=plot_progress)
    #save([fireflies], "training_set", append_datetime=True)
    #trajectories, fireflies = game.generate_trajectories(100, distance=10)
    trajectories, fireflies, _ = generate_trajectories(network, 10, max_distance=10)
    plot_trajectories(trajectories, fireflies)
    #game.network.print_weights()
    #save(session.run(game.network.weights), "bs100")
    #plot_weight_histograms(networks)
    #save(networks, "100networks-nodis-sgd")
    #networks = load("nodis_sgd/2017-06-12.14-50-14/"
    #networks = load("100networks-nodis-sgd-2017-06-12-145623.pkl")
    #networks, num_trials = train_repeatedly(session, n=100, max_trials=2500,
    #                                        batch_size=1)
    #returns = generate_trajectories(networks, num_trajectories=1, distance=10)
    #trajectories, fireflies, final_distances = returns
    #save([networks, num_trials, trajectories, fireflies, final_distances],
    #     "trajectories")
