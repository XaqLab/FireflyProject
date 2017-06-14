""" Only import packages needed for training the network. Network training will
sometimes be done on a remote server so no matplotlib, etc. """
import numpy as np
from numpy.random import rand, randn
import tensorflow as tf
import pickle
import signal
import datetime as dt
#from firefly_task import FireflyTask
from firefly_task_rect import FireflyTask
from visualization import load


# default to no plots for batch training on server
plot_progress = False
import __builtin__
if "__IPYTHON__" in vars(__builtin__):
    # plot progress for interactive training in iPython
    from visualization import *


def save(stuff, name, append_datetime=False):
    """ Save stuff to a file. """
    filename = name
    if append_datetime:
        date_and_time = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename += "-" + date_and_time
    filename += ".pkl"
    with open(filename, "w") as filehandle:
        pickle.dump(stuff, filehandle)


def train_repeatedly(tf_session, n, max_trials, batch_size=1):
    """ Train the network n times and save the weights. Each weight can then be
    plotted as a histogram to see how robust the training method is. """
    game = FireflyTask(tf_session)
    weights = []
    num_trials = []
    for i in range(n):
        trials, _ = game.practice(batch_size=batch_size, max_trials=max_trials,
                                  distance=1)
        weights.append(game.tf_session.run(game.network.weights))
        num_trials.append(trials)
        game = FireflyTask(tf_session)
    return weights, num_trials


def generate_trajectories(networks, num_trajectories, distance):
    """ Generate num_trajectories for each network in networks. """
    session = tf.Session()
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
    print "Let's eat some fireflies, yeah!"
    session = tf.Session()
    game = FireflyTask(session)
    game.practice(batch_size=1, max_trials=2000, distance=1,
                  firefly_file=None, plot_progress=plot_progress)
    #hand_picked = [np.array([[ 0.63661977, -0.00000000],
                             #[-0.00000000, -1.00000000],
                             #[-2.20000000, -0.50000000]], dtype=np.float32)]
    #hand_picked = [np.array([[ 0.68892437, -0.00465345],
                             #[-0.21578074, -1.02819788],
                             #[-1.98275447, -0.50836277]], dtype=np.float32)]
    #hand_picked = [np.array([[ 1.0, 0.0],
                             #[ 0.0, 1.0],
                             #[ 0.0, 0.0]], dtype=np.float32)]
    #game.network.set_weights(hand_picked)
    #trajectories, fireflies = game.generate_trajectories(100, distance=10)
    plot_trajectories(game, *game.generate_trajectories(10, distance=10)[:2])
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
