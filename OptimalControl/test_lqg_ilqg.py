"""
This file contains the unit tests for kalman_lqg.py which is a port of
kalman_lqg.m obtained from Emo Todorov's website at the University of
Washington.
"""

from __future__ import print_function
import matlab.engine
import numpy as np
from statistics import mean, stdev
from scipy.linalg import svd, norm
import scipy.stats
from numpy.random import randn
#from bokeh.plotting import figure, output_file, gridplot, vplot, show
import ipdb
import pickle
import time
import random
from optimal_control import *
from kalman_lqg import kalman_lqg, matlab_kalman_lqg
from iLQG import iterative_lqg


def perturb(matrix_trajectory, scale):
    """ Perturb the given matrix trajectory with Gaussian noise scaled by
    scale.  """
    perturbed_trajectory = matrix_trajectory.flatten()
    for i in range(len(perturbed_trajectory)):
        perturbed_trajectory[i] += scale*randn()
    return perturbed_trajectory.reshape(matrix_trajectory.shape)


def generate_kalman_lqg_LTI_regression_tests(filename, number_of_test_cases,
                                             time_samples):
    """
    5/27/2016 James Bridgewater
    I'm writing this function to generate test cases by picking random linear
    time-invariant systems and quadratic cost functions along with control and
    state dependent noise that fit into the framework from Emo Todorov's
    2005 Neural Computation paper.  I pass these to the MATLAB code he
    published to get the state esimation and feedback control matrices and save
    them along with the system description for use in regression tests that
    I will be using on the python code I'm developing as a behavioral model for
    animals performing foraging tasks.  This code will start out as a straight
    port of Todorov's code and then be extended for use with time-varying and
    non-linear cases.
    """
    test_cases = []
    eng = matlab.engine.start_matlab()
    test_case_number = 1
    while test_case_number <= number_of_test_cases:
        print("Test case #: ", test_case_number)
        system = random_kalman_lqg_LTI_system(time_samples)
        #function [K,L,Cost,Xa,XSim,CostSim,iter] = ...
        K, L, Cost, Xa, XSim, CostSim, iterations = \
                matlab_kalman_lqg(eng, system)
        if iterations < 500:
            """ Keep the results if the algorithm converged before stopping at
            the maximum number of iterations. """
            solution = {'K': K, 'L': L, 'Cost': Cost,
                        'Xa': Xa, 'iterations':iterations}
            test_case = {'system': system, 'solution': solution}
            test_cases.append(test_case)
            test_case_number = test_case_number + 1
    
    # Save the test cases
    with open(filename, 'w') as file_handle:
        pickle.dump(test_cases, file_handle)
    eng.quit()
    

def make_small_LTI_test_set():
    """
    1/23/2017 James Bridgewater
    Creating this function to make a small LTI test set using a subset of the 
    the existing 87 test set.
    """
    # Load the saved test cases
    with open("LTI_all.pkl", 'r') as file_handle:
        test_cases = pickle.load(file_handle)
    small_test_set = []
    # pick ten random numbers from 0 to 86 without replacement
    all_test_numbers = range(len(test_cases))
    test_numbers = random.sample(all_test_numbers, 10)
    for test_number in test_numbers:
        #print("Test case #: ", test_number)
        small_test_set.append(test_cases[test_number])
    # Save the test cases
    with open("LTI_regression.pkl", 'w') as file_handle:
        pickle.dump(small_test_set, file_handle)
    

def generate_kalman_lqg_LTV_regression_tests(filename, number_of_test_cases,    
                                             time_samples, DEBUG=False):
    """
    The idea here is to generate a random linear time-varying LQG
    system along with a random initial state and test the feedback and
    filter matrices returned by the iterative algorithm in kalman_lqg.py by
    perturbing them and making sure that the perturbed matrices always produce
    higher control costs and estimation errors than the originals.  This
    provides strong evidence that the algorithm in kalman_lqg.py is returning
    the optimal solution.  Once I believe the code is producing the optimal
    result for a test case it is added to the regression tests.
    """
    test_cases = []
    test_case_number = 1
    while test_case_number <= number_of_test_cases:
        if DEBUG:
            system = load_kalman_lqg_system()
            print("System loaded for debugging")
            ipdb.set_trace()
        else:
            system = random_kalman_lqg_LTV_system(time_samples)
            print("System generated")
    
        #############################################################
        # Call kalman_lqg
        #############################################################
        
        #K,L,Cost,Xa,XSim,CostSim = \
        return_values = kalman_lqg(system)
        K = np.array(return_values[0])
        L = np.array(return_values[1])
        Cost = np.array(return_values[2])
        Xa = np.array(return_values[3])
        XSim = np.array(return_values[4])
        CostSim = np.array(return_values[5])
        # I added this return value to make sure the algorithm converged
        iterations = np.array(return_values[6])
        if iterations == 500:
            # didn't converge, hit max iterations try another system
            continue
        else:
            test_case_number = test_case_number + 1
            print("System solved")
        
        """
        Calculate NSim from the difference between the means and the larger of the
        two mean estimate uncertainties.
        """
        NSim = 100 # number of simulations per batch
        timed_out = False # define so if timed_out doesn't fail on first loop
        N = 0 # define N so if N == NSim doesn't fail on first loop
        perturbation_size = 0.1
        # see if suboptimal matrices fail
        #K = perturb(K, 2*perturbation_size)
        #L = perturb(L, 2*perturbation_size)
        perturbation = 0
        costs_too_large = False
        while perturbation < 40 and not costs_too_large:
            # perturb the state trajectory and make sure the expected cost is higher
            if timed_out:
                # increase perturbation size
                perturbation_size *= 2
                print("perturbation_size", perturbation_size)
            if N == NSim:
                # decrease perturbation size
                perturbation_size *= 0.5
                print("perturbation_size", perturbation_size)
            suboptimal_K = perturb(K, perturbation_size)
            suboptimal_L = perturb(L, perturbation_size)
            N = 0
            optimal_costs = []
            suboptimal_costs = []
            time0 = time.time()
            timed_out = False
            confidence_level_met = False
            while not confidence_level_met and not timed_out:
                costs = compute_cost(system, K, L, NSim=NSim)
                optimal_costs = np.concatenate((optimal_costs, costs))
                optimal_cost = mean(optimal_costs)
                optimal_std = stdev(optimal_costs)
                costs = compute_cost(system, suboptimal_K, suboptimal_L,
                                     NSim=NSim)
                suboptimal_costs = np.concatenate((suboptimal_costs, costs))
                try:
                    suboptimal_cost = mean(suboptimal_costs)
                    suboptimal_std = stdev(suboptimal_costs)
                except (OverflowError, AttributeError):
                    # Catch errors caused by costs becoming very large as
                    # perturbation_size is increased in the hopes of meeting
                    # the confidence criteria.
                    costs_too_large = True
                    # try another system
                    break
                N = N + NSim
                print("N = ", N)
                print("optimal cost: ", optimal_cost,
                      "optimal std: ", optimal_std)
                print("suboptimal cost: ", suboptimal_cost,
                      "suboptimal std: ", suboptimal_std)
                # Construct a random variable that is the difference between the
                # uncertain mean costs and calculate the probability that this
                # random variable is less than zero. This corresponds to the
                # probability that the means are deceiving us as to which
                # trajectory produces the lower expected cost.
                difference_in_means = abs(optimal_cost - suboptimal_cost)
                std_of_diff = np.sqrt((optimal_std**2 + suboptimal_std**2)/N)
                prob_diff_is_less_than_zero = \
                        scipy.stats.t(N-1).cdf(-difference_in_means/std_of_diff)
                        #scipy.stats.norm(difference_in_means,std_of_diff).cdf(0)
                print(prob_diff_is_less_than_zero)
                if prob_diff_is_less_than_zero < 1e-3:
                    confidence_level_met = True
                if time.time() - time0 > 20: # seconds
                    timed_out = True
            if confidence_level_met:
                perturbation = perturbation + 1
                print("perturbation #%d" % perturbation)
                if suboptimal_cost < optimal_cost:
                    """ Something is broken, save the system so we can debug it. """
                    save_kalman_lqg_system(system)
                    print("The state trajectory is not optimal!")
                    #print("Perturbation at L[%d,%d,%d]" % (row, col, t))
                    print("Suboptimal cost - optimal cost: %.15f" %
                          (suboptimal_cost - optimal_cost))
                assert suboptimal_cost >= optimal_cost
        # save test cases that pass
        solution = {'K': K, 'L': L, 'Cost': Cost, 'Xa': Xa}
        test_case = {'system': system, 'solution': solution}
        test_cases.append(test_case)
        print("Finished with test case #: ", test_case_number)

    with open(filename, 'w') as file_handle:
        pickle.dump(test_cases, file_handle)


def kalman_lqg_system_to_ilqg_system(system):
    """ Take a kalman_lqg system and produce a representation of it that can be
    passed to iterative_lqg. """
    assert norm(system['E0']) == 0, "can't add noise to state estimate in iLQG"
    assert len(system['Q'].shape) == 3, "Q must have 3 dimensions"
    x0 = system['X1']
    S0 = system['S1']
    # dimensions
    N = system['Q'].shape[2]    # number of time steps + 1
    nx = system['A'].shape[0]   # number of state variables
    nu = system['B'].shape[1]   # number of control inputs
    nw = system['C0'].shape[1]  # number of process noise variables
    ny = system['H'].shape[0]   # number of observable outputs
    nv = system['D0'].shape[1]  # number of observation noise variables
    def f(x, u, k):
        """
        The function f defines the deterministic system dynamics.
        If x(k+1) = Ax(k) + Bu(k) then f(x,u) = (A-I)x(k) + Bu(k).
        The argument k allows the A and B matrices to be time dependent.
        """
        I = np.identity(nx)
        A = get_matrix(system['A'], k)
        B = get_matrix(system['B'], k)
        return (A-I).dot(x) + B.dot(u) 
    def F(x, u, k):
        """ F produces the state and control dependent process noise. """
        B = get_matrix(system['B'], k)
        C0 = get_matrix(system['C0'], k)
        C = get_tensor(system['C'], k)
        Cu = np.zeros([nx, C.shape[2]])
        for i in range(C.shape[2]):
            # build Cu one column at a time
            Cu[:,i] = B.dot(C[:,:,i]).dot(u)
        covariance_C_effctive = C0.dot(C0.T) + Cu.dot(Cu.T)
        [u,s,v] = svd(covariance_C_effctive)
        C_effective = u.dot(np.diag(np.sqrt(s))).dot(v.T)
        return C_effective
    def g(x, u, k):
        """ The function g produces the observable outputs. """
        H = get_matrix(system['H'], k)
        return H.dot(x)
    def G(x, u, k):
        """ G produces the state and control dependent observation noise. """
        D0 = get_matrix(system['D0'], k)
        D = get_tensor(system['D'], k)
        Dx = np.zeros([ny, D.shape[2]])
        for i in range(D.shape[2]):
            # build Dx one column at a time
            Dx[:,i] = D[:,:,i].dot(x)
        covariance_D_effctive = D0.dot(D0.T) + Dx.dot(Dx.T)
        [u,s,v] = svd(covariance_D_effctive)
        D_effective = u.dot(np.diag(np.sqrt(s))).dot(v.T)
        return D_effective
    def h(x):
        """ The function h is the final state cost. """
        Qf = system['Q'][:,:,-1]
        return x.dot(Qf).dot(x)
    def l(x, u, k):
        """ The function l is the sum of incremental state and control costs. """
        Q = system['Q'][:,:,k]
        R = get_matrix(system['R'], k)
        return x.dot(Q).dot(x) + u.dot(R).dot(u)
    xf = np.zeros([nx])
    x_n0 = initial_state_trajectory(f, x0, xf, nu, N)
    return f, F, g, G, h, l, nu, x_n0, S0


def call_ilqg_with_kalman_lqg_system(system):
    return iterative_lqg(*kalman_lqg_system_to_ilqg_system(system))


def compare(value, expected_value, tolerance, name):
    """ A simple function that compares two arrays and raises an assertion
    error if they are not equivalent. """
    if (isinstance(value, np.ndarray)
        and isinstance(expected_value, np.ndarray)):
        flat_v = value.flatten()
        flat_ev = expected_value.flatten()
        for i in range(len(flat_ev)):
            if abs(flat_v[i] - flat_ev[i]) > abs(tolerance*flat_ev[i]):
                print(flat_ev[i])
                print(flat_v[i])
            assert abs(flat_v[i] - flat_ev[i]) < abs(tolerance*flat_ev[i]), \
                    name + " is not within tolerance"
    else:
        # assume values are scalars
        if abs(value - expected_value) > abs(tolerance*expected_value):
            print(expected_value)
            print(value)
        assert abs(value - expected_value) < abs(tolerance*expected_value), \
                name + " is not within tolerance"


def regression(test_case, algorithm, tolerance, max_samples=1000000):
    """ Compare the costs of the trajectories produced using the solution
    found by algorithm with those of the trajectories produced using the
    known solution provided in test_case. """
    test_system = test_case['system']
    test_solution = test_case['solution']
    solution = algorithm(test_system)
    delta_n = 100
    n = delta_n
    cost = mean(compute_cost(test_system, solution, delta_n))
    test_cost = mean(compute_cost(test_system, test_solution, delta_n))
    while abs(cost - test_cost)/test_cost > tolerance and n < max_samples:
        n += delta_n
        cost = ((1-float(delta_n)/n)*cost
                + sum(compute_cost(test_system, solution, delta_n))/n)
        test_cost = ((1-float(delta_n)/n)*cost
                     + sum(compute_cost(test_system,
                                        test_solution, delta_n))/n)
    print(n, cost, test_cost, cost - test_cost)
    if n >= max_samples:
        print("Reached maximum number of samples.")
    compare(cost, test_cost, tolerance, "stochastic cost")


def test_kalman_lqg_time_invariant():
    """ Run one time-invariant regression test using kalman_lqg. """
    filename = "LTI_regression.pkl"
    with open(filename, 'r') as file_handle:
        test_cases = pickle.load(file_handle)
    test_case = random.choice(test_cases)
    regression(test_case, kalman_lqg, 1e-3)


def test_kalman_lqg_time_varying():
    """ Run one time-varying regression test using kalman_lqg. """
    filename = "LTV_regression.pkl"
    with open(filename, 'r') as file_handle:
        test_cases = pickle.load(file_handle)
    test_case = random.choice(test_cases)
    regression(test_case, kalman_lqg, 1e-3)


def test_iterative_lqg_time_invariant():
    """ Run one time-invariant regression test using iterative_lqg. """
    filename = "LTI_regression.pkl"
    with open(filename, 'r') as file_handle:
        test_cases = pickle.load(file_handle)
    test_case = random.choice(test_cases)
    regression(test_case, call_ilqg_with_kalman_lqg_system, 1e-3)
    # This test passed on the test set below, but took a very long time because
    # test case #7 required 1589 iterations! And these test cases only have 10
    # time samples.
    #regression("ilqg_debug.pkl", call_ilqg_with_kalman_lqg_system, 1e-3)


def test_iterative_lqg_time_varying():
    """ Run one time-varying regression test using iterative_lqg. """
    filename = "LTV_regression.pkl"
    with open(filename, 'r') as file_handle:
        test_cases = pickle.load(file_handle)
    test_case = random.choice(test_cases)
    regression(test_case, call_ilqg_with_kalman_lqg_system, 1e-3)


if __name__ == "__main__":
    #make_small_LTI_test_set()
    # ilqg_debug4.pkl has no noise
    #generate_kalman_lqg_LTI_regression_tests("ilqg_debug4.pkl", 1, 3)
    # ilqg_debug3.pkl has only additive noise
    #generate_kalman_lqg_LTI_regression_tests("ilqg_debug3.pkl", 1, 3)
    #generate_kalman_lqg_LTI_regression_tests("ilqg_debug2.pkl", 1, 3)
    #generate_kalman_lqg_LTI_regression_tests("ilqg_debug.pkl", 10, 10)
    #generate_kalman_lqg_LTI_regression_tests("LTI_regression.pkl", 10, 100)
    #generate_kalman_lqg_LTV_regression_tests("LTV_regression.pkl", 10, 100)
    test_kalman_lqg_time_invariant()
    test_kalman_lqg_time_varying()
    test_iterative_lqg_time_invariant()
    test_iterative_lqg_time_varying()

