import numpy as np
import pickle
#from firefly_task import FireflyTask

#if "__IPYTHON__" in vars(__builtins__).keys():
if "__IPYTHON__" in __builtins__.keys():
    # plot progress for interactive training in iPython
    import matplotlib.pyplot as plt

# define some constants
pi = np.pi


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


def plot_trajectories(trajectories, fireflies):
    """ Plot a trajectory. """
    n = len(trajectories)
    if n == 1:
        colors = ["#%06x" % roygbiv(0, 256)]
    else:
        colors = ["#%06x" % roygbiv(i, n) for i in range(n)]
    figure = plt.figure()
    axes = plt.subplot(111)
    axes.set_aspect(1)
    #dim = game.network.dimensions
    #f_names = game.afun_names
    #axes.set_title("%s %s" % (dim, f_names))
    for i, trajectory in enumerate(trajectories):
        x = trajectory[:,0]
        y = trajectory[:,1]
        axes.plot(x, y, color=colors[i])
        firefly_x = fireflies[i][0][0]
        firefly_y = fireflies[i][0][1]
        axes.plot(firefly_x, firefly_y, color=colors[i], marker='o')
    plt.show(block=False)


def plot_objective_function(game):
    #w0 = game.eval(game.network.weights[0])
    #w0 = np.array([[2/pi, 0],[0, 1.0]])
    #w0 = np.array([[2/pi, 1.3],[1.6, 1.0]])
    w0 = np.array([[2.0, 0.6],[3.0, 1.0]])
    figure = plt.figure()
    axes = plt.subplot(111)
    game.firefly = np.array([[pi/4, 1.0]])
    dw = 0.1
    weight_range = np.arange(-4, 4+dw, dw)
    f = np.zeros(len(weight_range))
    new_weights = np.array(w0)
    for col in range(w0.shape[1]):
        for row in range(w0.shape[0]):
            weights = np.array(w0)
            for i, w in enumerate(weight_range):
                weights[row,col] = w
                game.network.set_weights(new_weights)
                f[i] = game.eval(game.objective)
            axes.plot(weight_range, f, label="%d, %d" % (row,col))
    handles, labels = axes.get_legend_handles_labels()
    axes.set_xlabel("Weight")
    axes.set_ylabel("Objective Function Value")
    axes.legend(handles, labels, ncol=2)
    plt.show(block=False)
    # reset weights to w0
    game.network.set_weights(w0)


def plot_network_response(game):
    """ Plot the response of the network to each input. """
    directions = np.arange(-pi/2, pi/2, 1e-2)
    distances = np.arange(0, 1, 1e-2)
    rotation = np.zeros(len(directions))
    step_size = np.zeros(len(distances))
    for i in range(len(directions)):
        network_input = np.array([[directions[i], 0.0]])
        rotation[i] = game.rotation.forward(input=network_input)
    for i in range(len(distances)):
        network_input = np.array([[0.0, distances[i]]])
        step_size[i] = game.step_size.forward(input=network_input)
    fig1 = plt.figure()
    upper_axes = plt.subplot(211)
    dim = game.network.dimensions
    f_names = game.afun_names
    upper_axes.set_title("%s %s" % (dim, f_names))
    upper_axes.set_ylabel("rotation")
    upper_axes.plot(directions, directions, linestyle="--")
    upper_axes.plot(directions, rotation)
    lower_axes = plt.subplot(212)
    lower_axes.set_ylabel("step size")
    lower_axes.plot(distances, distances, linestyle="--")
    lower_axes.plot(distances, step_size)
    plt.show(block=False)


def plot_weight_histograms(networks):
    """ Plot histograms of each weight in the network. """
    fig1 = plt.figure()
    axes = plt.subplot(111)
    dx = 0.1
    bins = np.arange(-10.0 + dx/2, 10, dx)
    x = (bins + dx/2)[:-1]
    num_networks = len(networks)
    num_layers = len(networks[0])
    for layer in range(num_layers):
        for i in range(networks[0][layer].shape[0]):
            for j in range(networks[0][layer].shape[1]):
                wij = np.array([network[layer][i,j] for network in networks])
                counts, _ = np.histogram(wij, bins=bins)
                axes.plot(x, counts)
                #axes.plot(x, counts, label="%d, %d" % (i,j))
    #handles, labels = axes.get_legend_handles_labels()
    #axes.legend(handles, labels, ncol=2)
    plt.show(block=False)


def plot_distance_histogram(final_distances):
    """ Plot a histogram of the final distances to the fireflies. """
    fig1 = plt.figure()
    axes = plt.subplot(111)
    dmin = min(final_distances)
    dmax = max(final_distances)
    dx = (dmax - dmin)/100.0
    bins = np.arange(dmin - dx, dmax + 2*dx, dx)
    axes.hist(np.log10(final_distances), bins=np.log10(bins))
    axes.set_xlabel("log_10 Distance")
    axes.set_ylabel("Fireflies")
    #axes.plot(x, counts, label="%d, %d" % (i,j))
    #handles, labels = axes.get_legend_handles_labels()
    #axes.legend(handles, labels, ncol=2)
    plt.show(block=False)


def plot_trials_histogram(num_trials):
    """ Plot a histogram of the number of trials required to train the
    networks. """
    fig1 = plt.figure()
    axes = plt.subplot(111)
    dx = 100.0
    dmin = min(num_trials)
    dmax = max(num_trials)
    bins = np.arange(0, dmax + 2*dx, dx)
    axes.hist(num_trials, bins=bins)
    axes.set_xlabel("Trials")
    axes.set_ylabel("Networks")
    #axes.plot(x, counts, label="%d, %d" % (i,j))
    #handles, labels = axes.get_legend_handles_labels()
    #axes.legend(handles, labels, ncol=2)
    plt.show(block=False)


class AccumulateData(object):
    """ Accumulate data during training. """
    def __init__(self):
        self.distance = []
        self.gradients = []
        self.figure = plt.figure()
        axes1 = plt.subplot(211)
        axes2 = plt.subplot(212)
        plt.show(block=False)


    def append(self, datum):
        """ Append new data. """
        # append new data and call callback
        self.distance.append(datum['distance'])
        self.gradients.append(datum['gradients'])


    def plot(self, datum):
        """ Append new data and then plot all the data. """
        self.append(datum)
        axes1, axes2 = self.figure.get_axes()
        axes1.clear()
        axes2.clear()
        #dim = game.network.dimensions
        #f_names = game.afun_names
        #axes1.set_title("%s %s %d" % (dim, f_names, batch_size))
        axes1.set_ylabel("Distance")
        axes2.set_ylabel("|Gradients|")
        #for i in range(dw_means.shape[0]):
        #    x = range(1,trial)
        #    axes1.plot(x, dw_means[i,:trial-1] + dw_stds[i,:trial-1])
        #    axes1.plot(x, dw_means[i,:trial-1])
        #    axes1.plot(x, dw_means[i,:trial-1] - dw_stds[i,:trial-1])
        axes1.plot(self.distance)
        #import ipdb; ipdb.set_trace()
        abs_gradients = abs(np.array(self.gradients))
        axes2.plot(np.min(abs_gradients, axis=1), color="blue",
                   linestyle="dotted")
        axes2.plot(np.mean(abs_gradients, axis=1), color="blue")
        axes2.plot(np.max(abs_gradients, axis=1), color="blue",
                   linestyle="dotted")
        #axes2.set_ylim([0,5])
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


if __name__ == "__main__":
    #game.network.set_weights("bs100-2017-05-23-143300.pkl")
    # set weights to some hand picked value
    #hand_picked = [np.array([[2/pi, 0.0],
                             #[0.0, 1.0],
                             #[0.0, 0.5]], dtype=np.float32)]
    #game.network.set_weights(hand_picked)
    #game.plot_objective_function()
    #game.print_weights()
    #game.generate_trajectories(10, distance=10)
    #plot_trajectories(game)
    #game.plot_network_response()
    #networks = load("batch_size_exp/2017-06-08.14-05-23/"
    #                + "100networks-constant-input-2017-06-08-140816.pkl")
    #plot_weight_histograms(networks)
    #networks = load("batch_size_exp/2017-06-08.14-15-13/"
    #               + "100networks-no-constant-input-2017-06-08-141810.pkl")
    #plot_weight_histograms(networks)
    #networks = 100*[hand_picked]
    #trajectories, fireflies = game.generate_trajectories(100, distance=10)
    #plot_trajectories(trajectories, fireflies, game)
    #plot_weight_histograms(networks)
    returns = load("trajectories.pkl")
    networks, num_trials, trajectories, fireflies, final_distances = returns
    plot_distance_histogram(final_distances)
    plot_trials_histogram(num_trials)

