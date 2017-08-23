""" Only import packages needed for training the network. Network training will
sometimes be done on a remote server so no matplotlib, etc. """
import numpy as np
from numpy.random import rand, randn
import torch
import pickle
import signal
import datetime as dt
from neural_network import *
#from firefly_task import *
from firefly_task_rect import *
from closures import in_a_row


# default to no plots for batch training on server
plot_progress = False
if "__IPYTHON__" in __builtins__.keys():
    # plot progress for interactive training in iPython
    from visualization import *


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


class MyRNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        Network with GRU hidden layers and a linear output layer.
        """
        super(MyRNN, self).__init__()
        #self.GRU = torch.nn.GRU(D_in, H, num_layers)
        self.GRU = torch.nn.GRUCell(D_in, H)
        self.linear = torch.nn.Linear(H, D_out)


    def forward(self, x, hx):
        """
        Forward pass of the model.
        """
        hidden = self.GRU(x, hx)
        y_pred = self.linear(hidden)
        return y_pred


def xavier_initialization(network, normal=True):
    """ Initialize the network weights using the method published by Xavier
    Glorot et al. 2010. """
    for p in network.parameters():
        if len(p.data.size()) == 2:
            # weight parameters have 2 dimensions
            if normal == True:
                # normally distributed weights
                torch.nn.init.xavier_normal(p.data, gain=1)
                #p.data = torch.ones(p.data.shape) # for debugging gradients
            else:
                # uniformly distributed weights
                torch.nn.init.xavier_uniform(p.data, gain=1)
        elif len(p.data.size()) == 1:
            # bias parameters have 1 dimension
            p.data = torch.zeros(len(p.data))


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


def print_initial_gradients(network, optimizer, n=100):
    """ Compute the average initial gradients for a batch of n inputs. """
    optimizer.zero_grad()
    for i in range(n):
        agent = Variable(torch.zeros([1,2]))
        firefly = new_trial()
        hx = Variable(torch.randn([1,2]))
        agent = move(agent, firefly, network, hx)
        distance = torch.norm(firefly - agent)
        cost = distance.pow(2)
        cost.backward(retain_graph=True)
    gradients = [p.grad.data.numpy()/n for p in network.parameters()]
    #gradients = np.concatenate([g.flatten() for g in gradients])
    print("Initial gradients:")
    for g in gradients:
        print(g, "\n")
    print()


class Tanh_x_2(torch.nn.Tanh):
    def forward(self, x):
        return 2*super(Tanh_x_2, self).forward(x)


if __name__ == "__main__":
    print("Let's eat some fireflies, yeah!")
    #### create network ####

    Din = 2
    Dhidden = 2
    Dout = 2
    sequence_length = 10
    #network = MyRNN(Din, Dhidden, Dout)

    dimensions = [2, 10, 10, 10, 2]
    #dimensions = [2, 2]
    #scaled_tanh = lambda x: 1.7159*torch.tanh(2.0*x/3.0)
    #activation_functions = [scaled_tanh]
    #activation_functions = [torch.nn.ReLU()]
    #activation = torch.nn.Tanh()
    activation = Tanh_x_2()
    activation_functions = [activation]*(len(dimensions) - 1)

    network = create_network(dimensions, activation_functions)

    #### initialize network ####

    xavier_initialization(network)

    #initial_weights = session.run(game.network.weights)
    #save([initial_weights], "initial_weights", append_datetime=True)
    #weights = [np.array([[ 0.63661977,  0.00000000],
    #weights = [np.array([[ 1.0,  0.0],
                         #[ 0.0,  1.0]], dtype=np.float32)]
    #biases = [np.array([ 0.0,  0.5], dtype=np.float32)]
    #hand_picked = {'weights':weights, 'biases':biases}
    #set_parameters(network, hand_picked)

    #### train network ####
    learning_rate = 1e-2
    #optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    data = AccumulateData()

    print_parameters(network)
    print_initial_gradients(network, optimizer)

    trial = 0
    #max_trials = 0
    max_trials = 5000
    firefly = Variable(torch.ones([1,2]))
    agent = Variable(torch.zeros([1,2]))
    caught_in_a_row = in_a_row(caught, [0])
    with GracefulInterruptHandler() as h:
        while caught_in_a_row(firefly, agent) < 100 and trial < max_trials:
            trial = trial + 1
            firefly = new_trial()
            agent = Variable(torch.zeros([1,2]))
            distance = torch.norm(firefly - agent)
            #hx = Variable(torch.zeros([1,2]))
            hx = Variable(torch.randn([1,2]))
            step = 0
            while (not caught(firefly, agent) and
                   (step == 0 or distance_change < 0)):
                step = step + 1
                if step % 100 == 0:
                    print("Step = ", step)
                agent = move(agent, firefly, network, hx)
                new_distance = torch.norm(firefly - agent)
                distance_change = (new_distance - distance).data[0]
                distance = new_distance
                cost = new_distance.pow(2)
                optimizer.zero_grad()
                cost.backward(retain_graph=True)
                optimizer.step()
            gradients = [p.grad.data.numpy() for p in network.parameters()]
            gradients = np.concatenate([g.flatten() for g in gradients])
            datum = {'distance':new_distance.data[0], 'gradients':gradients}
            if trial % 100 == 0:
            #if trial % 5 == 0:
                print("\n", trial, new_distance.data[0])
                data.plot(datum)
            else:
                data.append(datum)
            if h.interrupted:
                # Allow user to stop training using Control-C.
                break
            # add option to save all training data
            # add option to manually adjust learning rate while training
            #if trial % 100 == 0:
                #import ipdb; ipdb.set_trace()

    #print_parameters(network)

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
