"""
This is an implementation of the optimal control algorithm from a 2007 paper by
Li and Todorov titled "Iterative linearization methods for approximately
optimal control and estimation of non-linear stochastic system." Unlike that
paper, this implementation uses Todorov's 2005 code (kalman_lqg) as the inner
loop in this iterative algorithm. In general the system to be controlled will
have non-linear dynamics, non-quadratic costs, and control and state dependent
noise. An initial state trajectory is chosen and the control trajectory
required to produce that state trajectory is computed.  At each time step, the
system dynamics will be linearized, the costs quadratized, the process noise
linearized with respect to the control signal and the observation noise
linearized with respect to the state so that kalman_lqg can be used to find an
approximately optimal control law. This approximate control law is used to
update the state and control trajectories used for the linearization and
quadritization process and this two-step process is repeated until the solution
reaches a steady state.
"""

from numpy import array, zeros, ones, identity, swapaxes, einsum, diag
from numpy import sqrt
from numpy.linalg import pinv
from numpy.random import randn
from numdifftools import Jacobian, Hessian
from optimal_control import *
from kalman_lqg import kalman_lqg
# debug stuff
import sys, re
from bokeh.plotting import figure, output_file, gridplot, show


def compute_control_trajectory(f, x, nu):
    """ Compute the control trajectory from the state trajectory and
    the function that describes the system dynamics. """
    assert len(x.shape) == 2, "x must have 2 dimensions"
    # Allocate memory for the control trajectory.
    N = x.shape[1]
    u = zeros([nu, N-1])
    # Calculate the the control input estimate, u_hat, for the first time step.
    u_hat = zeros(nu)
    dt = 1.0  # until there's a reason to use something else
    dx = x[:,1] - x[:,0]
    dfdu = Jacobian(lambda u: f(x[:,0], u, 0))
    u_hat = pinv(dfdu(u_hat)).dot(dx/dt - f(x[:,0], u_hat, 0))
    for k in range(N-1):
        dfdu = Jacobian(lambda u: f(x[:,k], u, k))
        # find the change in u that makes f(x,u,k)dt close to dx
        dx = x[:,k+1] - x[:,k]
        du = pinv(dfdu(u_hat)).dot(dx/dt - f(x[:,k], u_hat, k))
        u_hat += du
        u[:,k] += u_hat
    return u


def linearize_and_quadratize(f, F, g, G, h, l, x, u, S0, derivatives=None):
    """ Linearize the system dynamics and quadratize the costs around the state
    and control trajectories described by x and u for a system governed by the
    following equations.
        dx = f(x,u)dt + F(x,u)dw(t)
        dy = g(x,u)dt + G(x,u)dv(t)
        J(x) = E(h(x(T)) + integral over t from 0 to T of l(t,x,u))
    Where J(x) is the cost to go.
    We are using kalman_lqg as the 'inner loop' of this algorithm and it does
    not explicitly support linear state and control costs so we will augment
    the state vector to include the control inputs and a constant term. The
    augmented state vector, xa, is shown below.

    xa[k] = (x[k] u[k-1] 1).T

    This requires augmentation of the matrices A, B, C0, H, D, and Q. The
    augmentation of C0, H, and D is trivial as it simply involves adding zeros.
    The augmentation of A contains an additional 1 for the constant term added
    to the state vector.  The augmentation of B contains an identity submatrix
    which enables the addition of the control inputs to the state vector. The
    augmentation of Q is the most interesting.

    Qa[k] = [[ Q[k]    0         q[k]/2   ]
             [ 0       R[k-1]    r[k-1]/2 ]
             [ q[k]/2  r[k-1]/2  qs[k]    ]]
    
    Since the control costs are now state costs the control costs passed to
    kalman_lqg are zero, i.e. Ra = 0.
    """
    dt = 1.0  # until there's a reason to use something else
    nx = x.shape[0]
    nxa = x.shape[0] + u.shape[0] + 1 # for state augmentation
    nu = u.shape[0]
    szC0 = F(x[:,0], u[:,0], 0).shape[1]
    ny = g(x[:,0], u[:,0], 0).shape[0]
    szD0 = G(x[:,0], u[:,0], 0).shape[1]
    N = x.shape[1]
    system = {}

    # build the vector for the initial augmented state
    x0a = [0.0 for i in range(nx)]
    u0a = [0.0 if i != nu else 1.0 for i in range(nu+1)]
    system['X1'] = array(x0a + u0a)
    system['S1'] = zeros([nxa,nxa])
    system['S1'][0:nx,0:nx] = S0
    system['A'] = zeros([nxa, nxa, N-1])
    system['B'] = zeros([nxa, nu, N-1])
    system['C0'] = zeros([nxa, szC0, N-1])
    system['C'] = zeros([nu, nu, szC0, N-1])
    system['H'] = zeros([ny, nxa, N])
    system['D0'] = zeros([ny, szD0, N])
    system['D'] = zeros([ny, nxa, szD0, N])
    system['Q'] = zeros([nxa, nxa, N])
    system['R'] = zeros([nu, nu, N-1])
    for k in range(N-1):
        if derivatives == None:
            dfdx = Jacobian(lambda x: f(x, u[:,k], k))
            dfdu = Jacobian(lambda u: f(x[:,k], u, k))
            dFdu = Jacobian(lambda u: F(x[:,k], u, k))
            dgdx = Jacobian(lambda x: g(x, u[:,k], k))
            dGdx = Jacobian(lambda x: G(x, u[:,k], k))
            dldx = Jacobian(lambda x: l(x, u[:,k], k))
            d2ldx2 = Hessian(lambda x: l(x, u[:,k], k))
        else:
            dfdx = derivatives['dfdx']
            dfdu = derivatives['dfdu']
            dFdu = derivatives['dFdu']
            dgdx = derivatives['dgdx']
            dGdx = derivatives['dGdx']
            dldx = derivatives['dldx']
            d2ldx2 = derivatives['d2ldx2']
        A = dfdx(x[:,k]) + identity(nx)
        B = dfdu(u[:,k])
        C0 = sqrt(dt)*F(x[:,k], u[:,k], k)
        # dFdu is x by w by u, BC is x by u by w, so swap last 2 dimensions
        # and multiply by pinv(B) to get C
        C = sqrt(dt)*einsum('hi,ijk', pinv(B), swapaxes(dFdu(u[:,k]), -1, -2))
        H = dgdx(x[:,k])
        system['D0'][:,:,k] = G(x[:,k], u[:,k], k)/sqrt(dt)
        # dGdx is y by v by x, D is y by x by v, so swap last 2 dimensions
        #D = swapaxes(dGdx(x[:,k]), -1, -2)/sqrt(dt)
        D = dGdx(x[:,k])/sqrt(dt)
        # State cost, constant, linear, quadratic terms
        qs = dt*l(x[:,k], u[:,k], k)
        q = dt*dldx(x[:,k])
        Q = dt*d2ldx2(x[:,k])
        if k == 0:
            # Due to state augmentation, the cost for control at k=0 will be
            # paid when k=1 so r[0] and R[0] are all zeros.
            r = zeros(nu)
            R = zeros([nu, nu])
        else:
            dldu = Jacobian(lambda u: l(x[:,k-1], u, k-1))
            d2ldu2 = Hessian(lambda u: l(x[:,k-1], u, k-1))
            r = dt*dldu(u[:,k-1])
            R = dt*d2ldu2(u[:,k-1])
        # augment matrices to accommodate linear state and control costs
        Aa = zeros([nxa, nxa])
        Aa[0:nx,0:nx] = A
        Aa[-1,-1] = 1.0
        system['A'][:,:,k] = Aa
        Ba = zeros([nxa, nu])
        Ba[0:nx,0:nu] = B
        Ba[nx:nx+nu,0:nu] = identity(nu)
        system['B'][:,:,k] = Ba
        C0a = zeros([nxa, szC0])
        C0a[0:nx,0:szC0] = C0
        system['C0'][:,:,k] = C0a
        Ha = zeros([ny, nxa])
        Ha[0:ny,0:nx] = H
        system['H'][:,:,k] = Ha
        for j in range(D.shape[2]):
            Da = zeros([ny, nxa])
            Da[0:ny,0:nx] = D[:,:,j]
            system['D'][:,:,j,k] = Da
        Qa = zeros([nxa, nxa])
        Qa[0:nx,0:nx] = Q
        Qa[0:nx,nx+nu] = q/2
        Qa[nx+nu,0:nx] = q/2
        Qa[nx:nx+nu,nx:nx+nu] = R
        Qa[nx:nx+nu,nx+nu] = r/2
        Qa[nx+nu,nx:nx+nu] = r/2
        Qa[-1,-1] = qs
        system['Q'][:,:,k] = Qa
        # Control costs are built into the augmented Q matrix, Qa so R=0.
        system['R'][:,:,k] = zeros([nu, nu])

    # last time point
    H = dgdx(x[:,N-1])
    Ha = zeros([ny, nxa])
    Ha[0:ny,0:nx] = H
    system['H'][:,:,N-1] = Ha
    # not clear what u should be for the last time step
    #system['D0'][:,:,N-1] = G(x[:,N-1], u[:,N-2], k)/sqrt(dt)
    system['D0'][:,:,N-1] = G(x[:,N-1], zeros(nu), k)/sqrt(dt)
    D = dGdx(x[:,N-1])/sqrt(dt)
    for j in range(D.shape[2]):
        Da = zeros([ny, nxa])
        Da[0:ny,0:nx] = D[:,:,j]
        system['D'][:,:,j,N-1] = Da
    # last time point, use h(x) for the state cost 
    qs = h(x[:,N-1])
    dhdx = Jacobian(lambda x: h(x))
    q = dhdx(x[:,N-1])
    d2hdx2 = Hessian(lambda x: h(x))
    Q = d2hdx2(x[:,N-1])
    dldu = Jacobian(lambda u: l(x[:,N-2], u, N-2))
    r = dt*dldu(u[:,N-2])
    d2ldu2 = Hessian(lambda u: l(x[:,N-2], u, N-2))
    R = dt*d2ldu2(u[:,N-2])
    Qa = zeros([nxa, nxa])
    Qa[0:nx,0:nx] = Q
    Qa[0:nx,nx+nu] = q/2
    Qa[nx+nu,0:nx] = q/2
    Qa[nx:nx+nu,nx:nx+nu] = R
    Qa[nx:nx+nu,nx+nu] = r/2
    Qa[nx+nu,nx:nx+nu] = r/2
    Qa[-1,-1] = qs
    system['Q'][:,:,N-1] = Qa
    # iLQG does not accommodate noise added to the state estimate
    system['E0'] = zeros([1, 1, N-1])
    return system


def update_trajectories(f, x_n, u_n, La):
    """ Update the nominal state and control trajectories to use for
    linearizing and quadratizing the system dynamics and costs. """
    dt = 1.0  # until there's a reason to use something else
    N = La.shape[2] + 1
    nu = u_n.shape[0]
    u_p = zeros([nu, N-1])
    nx = x_n.shape[0]
    x_p = zeros([nx, N])
    x_p[:,0] = x_n[:,0]
    l = zeros([nu, N-1])
    L = zeros([nu, nx, N-1])
    Lu = zeros([nu, nu, N-1])
    for k in range(N-1):
        x = x_p[:,k] - x_n[:,k]
        # parse La to get L and l
        L[:,:,k] = La[:,0:nx,k]
        l[:,k] = La[:,-1,k]
        Lu[:,:,k] = La[:,nx:nx+nu,k]
        if Lu.any() != 0:
            # If Lu != 0 then the control input for the current time step
            # depends on the control input for the previous time step. This
            # is not allowed in the control law, u(k) = l(k) + L(k)x(k)
            print "Lu is not zero!"
            import ipdb; ipdb.set_trace()
        u = -l[:,k] - L[:,:,k].dot(x)
        u_p[:,k] = u_n[:,k] + u
        x_p[:,k+1] = x_p[:,k] + f(x_p[:,k], u_p[:,k], k)*dt
    return x_p, u_p, L, l


def compare_systems(system, previous_system):
    """ Used for debugging. """
    from numpy.linalg import norm
    tolerance = 1e-6
    N = system['R'].shape[2]
    if norm(system['X1'] - previous_system['X1']) > tolerance:
        print "X1 has changed"
    if norm(system['S1'] - previous_system['S1']) > tolerance:
        print "S1 has changed"
    for k in range(N):
        if norm(system['A'][:,:,k] - previous_system['A'][:,:,k]) > tolerance:
            print "A has changed"
            print previous_system['A'][:,:,k]
            print system['A'][:,:,k]
        if norm(system['B'][:,:,k] - previous_system['B'][:,:,k]) > tolerance:
            print "B has changed"
            print previous_system['B'][:,:,k]
            print system['B'][:,:,k]
        if norm(system['C0'][:,:,k] - previous_system['C0'][:,:,k]) > tolerance:
            print "C0 has changed"
            print previous_system['C0'][:,:,k]
            print system['C0'][:,:,k]
        if norm(system['H'][:,:,k] - previous_system['H'][:,:,k]) > tolerance:
            print "H has changed"
            print previous_system['H'][:,:,k]
            print system['H'][:,:,k]
        if norm(system['D0'][:,:,k] - previous_system['D0'][:,:,k]) > tolerance:
            print "D0 has changed"
            print previous_system['D0'][:,:,k]
            print system['D0'][:,:,k]
        if norm(system['Q'][:,:,k] - previous_system['Q'][:,:,k]) > tolerance:
            print "Q has changed at k =", k
            print system['Q'][:,:,k] - previous_system['Q'][:,:,k]
        if norm(system['R'][:,:,k] - previous_system['R'][:,:,k]) > tolerance:
            print "R has changed"
            print previous_system['R'][:,:,k]
            print system['R'][:,:,k]
    

def iterative_lqg(f, F, g, G, h, l, nu, x_n0, S0, derivatives=None):
    """ An implementation of Todorov's 2007 iterative LQG algorithm.  The
    system is described by these equations:
        dx = f(x,u)dt + F(x,u)dw(t)
        dy = g(x,u)dt + G(x,u)dv(t)
        J(x) = E(h(x(T)) + integral over t from 0 to T of l(t,x,u))
    Where T is N-1 times dt and J(x) is the cost to go.
    -> nu is the number of elements in the control vector u
    -> x_n0 is the initial state trajectory
    -> S0 is the initial state covariance matrix
    This algorithm returns state and control trajectories for a single run
    along with the state estimates and the filter and feedback matrices used to
    compute the state estimates and feedback controls.
    Because the system is, in general, non-linear the principal of certainty
    equivalence does not apply and K and L will change from run to run.
    """

    # start MATLAB engine, if using matlab_kalman_lqg
    #import matlab.engine
    #from test_kalman_lqg import matlab_kalman_lqg
    #eng = matlab.engine.start_matlab()

    N = x_n0.shape[1]   # the number of points on the state trajectory
    nx = x_n0.shape[0]  # the number of state variables
    x_n = x_n0
    u_n = compute_control_trajectory(f, x_n, nu)

    # Compute the cost of the initial trajectories.
    cost = trajectory_cost({'l':l, 'h':h}, x_n, u_n)
    print "iLQG initial trajectory cost:", cost
    # Linearize and quadratize around (x=x_n, u=u_n).
    system = linearize_and_quadratize(f, F, g, G, h, l, x_n, u_n, S0,
                                      derivatives)

    # save the initial system for debugging
    initial_system = system

    solution = kalman_lqg(system)
    K, L = solution['K'], solution['L']
    # try MATLAB code on this system to make sure it returns the same solution
    #K, L, Cost, Xa, XSim, CostSim, iterations = matlab_kalman_lqg(eng, system)

    has_not_converged = True
    iteration = 1
    while has_not_converged:

        if False:
            # plot some figures
            figures = []
            script_name = re.split(r"/",sys.argv[0])[-1]
            output_file_name = script_name.replace(".py", ".html")
            output_file(output_file_name, title="")
            
            # plot the state and control trajectories
            ku = range(N-1)
            kx = range(N)
            ps = figure(title="State trajectory", x_axis_label='time',
                        y_axis_label='')
            #p.line(x_n[0,:], x_n[1,:], line_width=2, line_color="blue")
            ps.line(kx, x_n[0,:], line_width=2, line_color="blue")
            ps.line(kx, x_n[1,:], line_width=2, line_color="green")
            
            # plot the control trajectory
            pc = figure(title="Control trajectory", x_axis_label='time',
                        y_axis_label='')
            #pc.line(u_n[0,:], u_n[1,:], line_width=2, line_color="blue")
            pc.line(ku, u_n[0,:], line_width=2, line_color="blue")
            pc.line(ku, u_n[1,:], line_width=2, line_color="green")
            figures.append([ps, pc])
            
            p = gridplot(figures)
            show(p)

        # Use the control policy from kalman_lqg to update the nominal state
        # and control trajectories.
        x_n, u_n, L_n, l_n = update_trajectories(f, x_n, u_n, L)
        previous_cost = cost
        cost = trajectory_cost({'l':l, 'h':h}, x_n, u_n)
        print "iLQG iteration %d trajectory cost: %.12f" % (iteration, cost)
        #if abs(cost / previous_cost - 1) < 1e-6:
        #if abs(cost - previous_cost) < 1e-12:
        if abs(cost - previous_cost) < 1e-6:
            # convergence criteria has been met, yay!
            has_not_converged = False
        else:
            # Re-linearize the system dynamics and re-quadratize the system
            # costs along the new nominal trajectories.
            system = linearize_and_quadratize(f, F, g, G, h, l, x_n, u_n, S0,
                                              derivatives)
            # Update the feedback control law.
            solution = kalman_lqg(system)
            K, L = solution['K'], solution['L']
            #K, L, Cost, Xa, XSim, CostSim, iterations = \
            #        matlab_kalman_lqg(eng, system)
        iteration = iteration + 1

    # compare final system to initial system for debugging
    #final_system = system
    #print "Comparing final and initial systems"
    #compare_systems(final_system, initial_system)

    # exit MATLAB engine, if using matlab_kalman_lqg
    #eng.quit()
    return {'x_n':x_n, 'u_n':u_n, 'L':L_n, 'l':l_n, 'K':K[0:nx,:],
            'system':system}


