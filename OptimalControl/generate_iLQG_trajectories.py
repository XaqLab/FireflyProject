import sys, re
import numpy as np
from numpy.random import randn
from numpy.linalg import cholesky, eig, pinv, norm, svd
from matplotlib import pyplot as plt
from iLQG import iterative_lqg
from kalman_lqg import kalman_lqg

# define constants
x0 = np.array([100.0,100.0])
S0 = np.identity(2)
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = 1.0/A
C0 = cholesky(np.array([[0.11, 0],[0, 0.11]]))
H = np.array([[1.3, 2.6], [9.0, 0.3]])
D0 = cholesky(np.array([[0.1, 0.0],[0.0, 0.1]]))
R = np.array([[0.3, 0.1],[0.1, 0.4]])
Q = np.array([[4.0, 1.0],[1.0, 3.0]])
Qf = np.array([[4.1, 1.0],[1.0, 3.2]])
# dimensions
nx = A.shape[0]
nu = B.shape[1]
nw = C0.shape[1]
ny = H.shape[0]
nv = D0.shape[1]

# define the system in iLQG form
def f(x, u):
    """ f defines the deterministic part of the system dynamics. """
    # If x(k+1) = Ax(k) + Bu(k) then f(x,u) = (A-I)x(k) + Bu(k)
    I = np.identity(nx)
    return (A-I).dot(x) + B.dot(u) 

def F(x, u):
    """ F defines the stochastic part of the system dynamics. """
    return C0

def g(x, u):
    """ g defines the deterministic part of the system observables. """
    return H.dot(x)

def G(x, u):
    """ G defines the stochastic part of the system observables. """
    return D0

def l(x, u):
    """ l defines the system costs prior to the final state. """
    return x.dot(Q).dot(x) + u.dot(R).dot(u)
    
def h(x):
    """ h defines the system costs in the final state. """
    return x.dot(Qf).dot(x)

def initial_state_trajectory(f, x0, xf, nu, N):
    """ Compute the initial state trajectory to use for generating stochastic
    state and control trajectories. Make the state trajectory a straight line
    from x0 to xf."""
    # Compute the straight line trajectory from x0 to xf.
    dx = (xf - x0) / float(N-1)
    x = np.array([x0 + i*dx for i in range(N)]).T
    return x

def generate_ilqg_trajectory(f, F, g, G, h, l, nu, x_n0, S0, derivatives=None):
    """ Generate an N point state trajectory and its corresponding N-1 point
    control trajectory for the opimal control system described by these
    equations:
        dx = f(x,u)dt + F(x,u)dw(t)
        dy = g(x,u)dt + G(x,u)dv(t)
        J(x) = E(h(x(T)) + integral over t from 0 to T of l(t,x,u))
    where x is a vector describing the state of the system, y is a vector
    containing measurable properties of the system and J is the cost to go.
    The argument nu is the number of elements in the control input vector u.
    The argument x_n0 is the initial nominal state trajectory.
    The argument S0 is the initial state covariance.
    The argument derivatives allows analytic derivatives to be passed in to
    speed up computation.
    """
    [u,s,v] = svd(S0)
    sqrt_S0 = u.dot(np.diag(np.sqrt(s))).dot(v.T)
    nx = x_n0.shape[0]
    N = x_n0.shape[1]
    u_n = np.zeros([nu,N-1])
    u_p = np.zeros([nu,N-1])
    x = np.zeros([nx,N])
    x_hat = np.zeros([nx,N])
    x_hat[:,0] = sqrt_S0.dot(randn(nx))
    x_n = x_n0
    x_p = np.zeros([nx,N])
    x_p[:,0] = x_n[:,0] + sqrt_S0.dot(randn(nx))
    x_p[:,0] = x_n[:,0]
    ny = len(g(x_n[:,0], u_n[:,0]))
    y_n = np.zeros([ny,N])
    y_p = np.zeros([ny,N])
    Lx = np.zeros([nu,nx,N-1])
    lx = np.zeros([nu,N-1])
    K = np.zeros([nx,ny,N-1])
    for k in range(N-1):
        #print "%3d" % k,
        #print norm(x_hat[:,k]), 0.1*norm(x_n[:,k])
        if k == 0 or norm(x_hat[:,k]) > 0.1*norm(x_n[:,k]):
            """
            If k=0 then we have not yet found an approximately optimal
            control law. If the estimated difference between the actual
            trajectory and nominal trajectory, x_hat, has deviated from the
            nominal trajectory, x_n, by more than 10% then update the control
            law.  Use x_hat to to determine when to recompute the nominal
            trajectory because using x_p - x_n is equivalent to making the
            system fully observable.
            """
            # Update x_n to reflect our current belief about the state of the
            # system before finding a new iLQG solution.
            x_n[:,k] += x_hat[:,k]
            solution = iterative_lqg(f, F, g, G, h, l, nu, x_n[:,k:], S0,
                                     derivatives)
            x_n[:,k:N] = solution[0]
            u_n[:,k:N-1] = solution[1]
            Lx[:,:,k:N-1] = solution[2]
            lx[:,k:N-1] = solution[3]
            K[:,:,k:N-1] = solution[4]
            system = solution[5]
            k_offset = k # used to index the time-varying matrices in system
            # calculate the nominal observations
            y_n[:,k:N-1] = np.array([g(x_n[:,j], u_n[:,j])
                                     for j in range(k,N-1)]).T
        y_n[:,N-1] = np.array(g(x_n[:,N-1], np.zeros(nu)))
        # calculate the control input
        u = -Lx[:,:,k].dot(x_hat[:,k]) - lx[:,k]
        u_p[:,k] = u + u_n[:,k]
        # calculate the next state
        A = system['A'][0:nx,0:nx,k-k_offset]
        B = system['B'][0:nx,:,k-k_offset]
        C0 = system['C0'][0:nx,:,k-k_offset]
        nw = C0.shape[1]
        #C = system['C'][0:nx,:,:,k-k_offset]
        x[:,k+1] = (A.dot(x[:,k]) + B.dot(u) + C0.dot(randn(nw)))
        x_p[:,k+1] = x[:,k+1] + x_n[:,k+1]
        # calculate the noisy observation
        H = system['H'][:,0:nx,k-k_offset]
        D0 = system['D0'][0:ny,:,k-k_offset]
        nv = D0.shape[1]
        #D = system['D'][0:nx,:,:,k-k_offset]
        y = H.dot(x[:,k]) + D0.dot(randn(nv))
        y_p[:,k] = y + y_n[:,k]
        # calculate the state estimate
        x_hat[:,k+1] = (A.dot(x_hat[:,k]) + B.dot(u)
                        + K[:,:,k].dot(y - H.dot(x_hat[:,k])))
        print x[:,k], x_hat[:,k]
        #print k, x_p[:,k], y_p[:,k]
    k = N-1
    print x[:,k], x_hat[:,k]
    #y = H.dot(x[:,k]) + D0.dot(randn(nv))
    #y_p[:,k] = y + y_n[:,k]
    #print k, x_p[:,k], y_p[:,k]
    return x_p, u_p, x, x_hat


N = 3 # number of time steps + 1
M = 9 # number of trajectories

# Find the analytic solution to this LQG system.
system = {}
system['X1'] = x0
system['S1'] = S0
system['A'] = A
system['B'] = B
system['C0'] = C0
system['C'] = np.zeros([nu, nu, 1])
system['H'] = H
system['D0'] = D0
system['D'] = np.zeros([ny, nx, 1])
system['E0'] = np.zeros([nx, 1])
system['Q'] = np.stack([Q if k < N-1 else Qf for k in range(N)], -1)
system['R'] = R

K_lqg, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(system, NSim=M)
#import matlab
#import matlab.engine
#from kalman_lqg import matlab_kalman_lqg
#eng = matlab.engine.start_matlab()
#K_lqg, L, Cost, Xa, XSim, CostSim, iterations = matlab_kalman_lqg(eng, system)
#eng.quit()

# Generate some iLQG trajectories
x = np.zeros([nx,M,N])
x_hat = np.zeros([nx,M,N])
x_p = np.zeros([nx,M,N])
u_p = np.zeros([nu,M,N-1])
x_n = np.zeros([nx,M,N])
u_n = np.zeros([nu,M,N-1])
Lx = np.zeros([nu,nx,M,N-1])
lx = np.zeros([nu,M,N-1])
# Generate the initial state trajectory.
xf = np.zeros([nx])
x_n0 = initial_state_trajectory(f, x0, xf, nu, N)
for m in range(M):
    x_p[:,m,:], u_p[:,m,:], x[:,m,:], x_hat[:,m,:] = generate_ilqg_trajectory(f,F,g,G,h,l,nu,x_n0,S0)

# plot the LQG state trajectories
kx = range(N)
p1 = plt.figure()
plt.title("LQG State Trajectories")
axes = plt.gca()
axes.set_xlabel('time')
for m in range(M):
    plt.plot(kx, XSim[0,m,:], linewidth=2, color="blue", linestyle='dotted',
             label="XSim[0]")
    plt.plot(kx, XSim[1,m,:], linewidth=2, color="green", linestyle='dotted',
             label="XSim[1]")
#plt.legend(loc="lower right")

# plot the iLQG state trajectories
kx = range(N)
p2 = plt.figure()
plt.title("iLQG State Trajectories")
axes = plt.gca()
axes.set_xlabel('time')
for m in range(M):
    plt.plot(kx, x_p[0,m,:], linewidth=2, color="blue", linestyle='dotted',
             label="x_p[0]")
    plt.plot(kx, x_p[1,m,:], linewidth=2, color="green", linestyle='dotted',
             label="x_p[1]")
#plt.legend(loc="lower right")

plt.show(block=False)

