import sys, re
import numpy as np
from numpy.random import randn
from numpy.linalg import cholesky, eig, pinv, norm
#from bokeh.plotting import figure, output_file, gridplot, show
from matplotlib import pyplot as plt
from iLQG import iterative_lqg
from kalman_lqg import kalman_lqg

# define constants
nx = 2
nu = 2
nw = 2
ny = 2
nv = 2
x0 = np.array([100.0,100.0])
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = 1.0/A
C0 = cholesky(np.array([[0.11, 0],[0, 0.11]]))
#C0 = np.zeros([nx, nw])
H = np.array([[5.0, 3.0], [2.0, 1.0]])
D0 = cholesky(np.array([[0.1, 0.0],[0.0, 0.1]]))
D0 = np.zeros([ny, nv])
R = np.array([[0.3, 0.1],[0.1, 0.4]])
Q = np.array([[4.0, 1.0],[1.0, 3.0]])
Qf = np.array([[4.1, 1.0],[1.0, 3.2]])

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

N = 10 # number of time steps + 1
M = 1 # number of trajectories

# Find the analytic solution to this LQG system.
system = {}
system['X1'] = x0
system['S1'] = np.identity(nx)
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
K, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(system)
u_lqg = np.zeros([nu,N-1])
x_lqg = np.zeros([nx,N])
x_lqg[:,0] = x0
for k in range(N-1):
    u_lqg[:,k] = -L[:,:,k].dot(x_lqg[:,k])
    x_lqg[:,k+1] = A.dot(x_lqg[:,k]) + B.dot(u_lqg[:,k])

# Generate some iLQG trajectories
x_p = np.zeros([nx,M,N])
u_p = np.zeros([nu,M,N-1])
for i in range(M):
    x_p[:,i,0] = x0 
dt = 1 # until there's a reason to use something else
x_n = np.zeros([nx,M,N])
u_n = np.zeros([nu,M,N-1])
Lx = np.zeros([nu,nx,M,N-1])
lx = np.zeros([nu,M,N-1])
for m in range(M):
    for k in range(N-1):
        print "%3d" % k,
        if k == 0 or norm(x_p[:,m,k] - x_n[:,m,k]) > 0.1*norm(x_n[:,m,k]):
            # If k=0 then we have not yet found an approximately optimal
            # control law. If the actual trajectory, x_p, has deviated from the
            # nominal trajectory, x_n, by more than 10% then update the control
            # law.
            solution = iterative_lqg(f, F, g, G, h, l, x_p[:,m,k], N-k)
            x_n[:,m,k:N] = solution[0]
            u_n[:,m,k:N-1] = solution[1]
            Lx[:,:,m,k:N-1] = solution[2]
            lx[:,m,k:N-1] = solution[3]
        # calculate the control input
        u_p[:,m,k] = (Lx[:,:,m,k].dot(x_p[:,m,k] - x_n[:,m,k]) + lx[:,m,k]
                      + u_n[:,m,k])
        # calculate the next state
        x_p[:,m,k+1] = (x_p[:,m,k] + f(x_p[:,m,k], u_p[:,m,k])*dt
                        + F(x_p[:,:,k], u_p[:,:,k]).dot(randn(nw)))

# for debugging purposes
#def print_array(x):
    #print "[",
    #for entry in x:
        #print "%11.6f" % entry,
    #print "]",
#np.set_printoptions(precision=6, suppress=True)

# print x_p - x_n
#print
#print "x_p - x_n"
#for k in range(N-1):
    #m = 0 # just need to look at one
    #print "%3d" % k,
    #print_array(x_p[:,m,k] - x_n[:,k])
    #print

# print the control inputs 
#print
#print "lx"
#for k in range(N-1):
    #m = 0 # just need to look at one
    #print "%3d" % k,
    #print_array(lx[:,k])
    #print

# setup output file to plot figures
#script_name = re.split(r"/",sys.argv[0])[-1]
#output_file_name = script_name.replace(".py", ".html")
#output_file(output_file_name, title="")

# plot the state trajectories
kx = range(N)
p1 = plt.figure()
plt.title("State Trajectories")
axes = plt.gca()
axes.set_xlabel('time')
plt.plot(kx, x_lqg[0,:], linewidth=2, color="blue",
        linestyle='dotted', label="x_lqg[0]")
plt.plot(kx, x_lqg[1,:], linewidth=2, color="green",
        linestyle='dotted', label="x_lqg[1]")
for m in range(M):
    #plt.plot(kx, x_n[0,m,:], linewidth=2, color="blue", label="x_n[0]")
    #plt.plot(kx, x_n[1,m,:], linewidth=2, color="green", label="x_n[1]")
    plt.plot(kx, x_p[0,m,:], linewidth=2, color="blue", linestyle='dashed',
             label="x_p[0]")
    plt.plot(kx, x_p[1,m,:], linewidth=2, color="green", linestyle='dashed',
             label="x_p[1]")
plt.legend(loc="lower right")

# plot the control trajectories
ku = range(N-1)
p2 = plt.figure()
plt.title("Control Trajectories")
axes = plt.gca()
axes.set_xlabel('time')
plt.plot(ku, u_lqg[0,:], linewidth=2, color="blue", linestyle='dotted',
         label="u_lqg[0]")
plt.plot(ku, u_lqg[1,:], linewidth=2, color="green", linestyle='dotted',
         label="u_lqg[1]")
for m in range(M):
    #plt.plot(ku, u_n[0,m,:], linewidth=2, color="blue", label="u_n[0]")
    #plt.plot(ku, u_n[1,m,:], linewidth=2, color="green", label="u_n[1]")
    plt.plot(ku, u_p[0,m,:], linewidth=2, color="blue", linestyle='dashed',
             label="u_p[0]")
    plt.plot(ku, u_p[1,m,:], linewidth=2, color="green", linestyle='dashed',
             label="u_p[1]")
plt.legend(loc="lower right")
plt.show(block=False)

## plot the state trajectories
#p3 = figure(title="State Trajectories", x_axis_label='state0',
#            y_axis_label='state1')
#p3.line(x_n[0,:], x_n[1,:], line_width=2, line_color="blue", legend="x_n")
#for m in range(M):
#    p3.line(x_p[0,m,:], x_p[1,m,:], line_width=2, line_color="blue",
#            line_dash='dashed', legend="x_p")
#p3.legend.location = "bottom_right"
#
## plot the control trajectories
#p4 = figure(title="Control Trajectories", x_axis_label='control0',
#            y_axis_label='control1')
#p4.line(u_n[0,:], u_n[1,:], line_width=2, line_color="blue", legend="u_n")
#for m in range(M):
#    p4.line(u_p[0,m,:], u_p[1,m,:], line_width=2, line_color="blue",
#            line_dash='dashed', legend="u_p")
#p4.legend.location = "bottom_right"
#
#p = gridplot([[p1, p2], [p3, p4]])
