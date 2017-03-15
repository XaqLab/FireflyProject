""" Shared optimal control functions. """

import numpy as np
from numpy.random import randn, rand, randint
from scipy.linalg import svd, norm, expm
from numpy.linalg import inv, eig, matrix_power


def matlab_matrix_string(M):
    """ Convert a Python list or array to a string in MATLAB matrix format,
    e.g. convert [[1,2],[3,4]] to '[1,2;3,4]' """
    matrix_string = "["
    for row in M:
        for entry in row:
            matrix_string += "%.15f" % entry + ", "
        matrix_string += "; "
    matrix_string += "];\n"
    return matrix_string


def print_args(system):
    """ Print out all the arguments passed to kalman_lqg so they can be cut and
    paste into MATLAB. """
    A = system['A']
    B = system['B']
    C0 = system['C0']
    C = system['C']
    H = system['H']
    D0 = system['D0']
    D = system['D']
    E0 = system['E0']
    X1 = system['X1']
    S1 = system['S1']
    R = system['R']
    Q = system['Q']
    print("A = ", matlab_matrix_string(A))
    print("B = ", matlab_matrix_string(B))
    for k in range(C.shape[2]):
        print("C(:,:,%d) = %s" % (k+1, matlab_matrix_string(C[:,:,k])))
    print("C0 = ", matlab_matrix_string(C0))
    print("H = ", matlab_matrix_string(H))
    for k in range(D.shape[2]):
        print("D(:,:,%d) = %s" % (k+1, matlab_matrix_string(D[:,:,k])))
    print("D0 = ", matlab_matrix_string(D0))
    print("E0 = ", matlab_matrix_string(E0))
    for k in range(Q.shape[2]):
        print("Q(:,:,%d) = %s" % (k+1, matlab_matrix_string(Q[:,:,k])))
    print("R = ", matlab_matrix_string(R))
    print("X1 = ", matlab_matrix_string(X1))
    print("S1 = ", matlab_matrix_string(S1))


def equal(A, B):
    """ Determine if two matrices are equal. """
    return norm(A - B) < 1e-12


def save_kalman_lqg_system(system, filename):
    """ Save the system dictionary passed to kalman_lqg so I can debug
    cases that fail. """
    with open(filename, 'w') as file_handle:
        pickle.dump(system, file_handle)


def load_kalman_lqg_system():
    """ Load the system dictionary for failure cases from a file they can be
    debugged. """
    with open("system.pkl", 'r') as file_handle:
        system = pickle.load(file_handle)
    return system


def random_symmetric_matrix(n):
    """ return a random n by n symmetric matrix """
    M = randn(n, n)
    return 0.5*(M + M.T)


def random_symmetric_positive_definite_matrix(n):
    """ return a random n by n positive definite matrix """
    eigenvalues = rand(n) + 0.1
    R = randn(n, n)
    # a symmetric matrix has orthogonal eigenvectors, the matrix exponential of
    # an antisymmetric matrix is an orthogonal matrix
    V = np.real(expm(R - R.T))
    positive_definite_matrix = V.dot(np.diag(eigenvalues)).dot(V.T)
    evals, evecs = eig(positive_definite_matrix)
    return positive_definite_matrix


def random_symmetric_positive_definite_matrices(n, N):
    """ Generate N random m by n non-singular matrices. """
    k = 0
    matrix = random_symmetric_positive_definite_matrix(n)
    matrices = [matrix]
    while k < N-1:
        while equal(matrix, matrices[-1]) or min(eig(matrix)[0]) < 0.1:
            # make sure matrix entries only change by a few percent each
            # time step and that the matrix is still positive definite
            evals, evecs = eig(matrix)
            matrix = matrices[-1] * (1 + 1e-2*random_symmetric_matrix(n))
        matrices.append(matrix)
        k = k + 1
    if N == 1:
        matrices = matrices[0]
    else:
        matrices = np.stack(matrices, -1)
    return matrices


def controllability_matrix(A,B):
    """ Calculate the controllability matrix.  """
    na,ma = A.shape
    assert na == ma, "A must be a square matrix"
    nb,mb = B.shape
    assert nb == na, "B must have the same number of rows as A"
    C = np.zeros([na, nb*mb])
    for i in range(na):
        C[:,i*mb:(i+1)*mb] = matrix_power(A,i).dot(B)
    return C


def observability_matrix(A,H):
    """ Calculate the observability matrix.  """
    na,ma = A.shape
    assert na == ma, "A must be a square matrix"
    nh,mh = H.shape
    assert mh == ma, "H must have the same number of columns as A"
    obs = np.zeros([nh*mh, na])
    for i in range(na):
        obs[i*nh:(i+1)*nh,:] = H.dot(matrix_power(A,i))
    return obs


def random_nonsingular_matrices(m, n, N=1):
    """ Generate N random m by n non-singular matrices. """
    k = 0
    matrix = randn(m, n)
    while min(svd(matrix)[1]) < 0.1:
        matrix = randn(m, n)
    matrices = [matrix]
    while k < N-1:
        while equal(matrix, matrices[-1]) or min(svd(matrix)[1]) < 0.1:
            # make sure matrix is not close to singular and only change each
            # entry by a few percent between time steps
            matrix = matrices[-1] * (1 + 1e-2*randn(m, n))
        matrices.append(matrix)
        k = k + 1
    if N == 1:
        matrices = matrices[0]
    else:
        matrices = np.stack(matrices, -1)
    return matrices


def random_nonsingular_matrix(m, n):
    """ Generate a random m by n non-singular matrix. """
    return random_nonsingular_matrices(m, n, 1)


def random_stable_state_transition_matrix(n):
    """ Generate a random CLOSED LOOP state transition matrix that has no
    eigenvalues greater than 1. """
    A = randn(n,n)
    while min(svd(A)[1]) < 0.1:
        A = randn(n,n)
    eigenvalues, eigenvectors = eig(A)
    # if any eigenvalues have positive real parts, multiply them by -1
    signs = 2*((np.real(eigenvalues) <= 0).astype(int) - 0.5)
    new_eigenvalues = signs * eigenvalues
    # produce a new matrix using the new eigenvalues and old eigenvectors
    new_A = eigenvectors.dot(np.diag(new_eigenvalues)).dot(inv(eigenvectors))
    # the new matrix may have very small imaginary parts, set them to zero
    new_A = np.real(new_A)
    F = expm(new_A)
    return F


def random_kalman_lqg_LTI_system(time_samples):
    """ Generate a random system for the optimal control algorithm implemented
    in Todorov's kalman_lqg.m MATLAB code. """
    N = time_samples    # number of discrete times in the state trajectory
    
    # number of state variables
    nx = randint(2,5)

    # number of control inputs
    nu = randint(2,5)

    # number of observable outputs
    ny = randint(2,5)

    # number of additive process noise variables
    np0 = randint(2,5)

    # number of control dependent process noise varaibles
    npc = randint(2,5)

    # number of additive measurement noise variables
    nm0 = randint(2,5)

    # number of state dependent measurement noise variables
    nms = randint(2,5)

    # number of internal noise variables
    ni = randint(2,5)

    # scale factor for noise matrices
    noise_scale = 1e-1

    """ generate a random linear, time invariant, open loop system
    x(k+1) = A*x(k) + B*u(k) """
    # system dynamics matrix, A
    A = random_nonsingular_matrix(nx, nx)

    # control input matrix, B
    B = random_nonsingular_matrix(nx, nu)

    # control input dependent noise matrices, C
    #C = noise_scale*randn(nu, nu, npc)
    C = np.zeros([nu, nu, npc])

    # additive process noise matrix, C0
    #C0 = noise_scale*randn(nx, np0)
    C0 = np.zeros([nx, np0])
    
    # measurement matrix, H 
    H = randn(ny, nx)

    # state dependent measurement noise matrices, D
    #D = noise_scale*randn(ny, nx, nms)
    D = np.zeros([ny, nx, nms])

    # additive measurement noise matrix, D0
    #D0 = noise_scale*randn(ny, nm0)
    D0 = np.zeros([ny, nm0])
    
    # internal noise that directly affects the state estimate
    # zero in LQG systems
    #E0 = noise_scale*randn(nx, ni)
    E0 = np.zeros([nx, ni])
    # pick a random initial state and initial covariance matrix
    X1 = randn(nx, 1)
    S1 = np.identity(nx) 

    # pick random state and control cost matrices
    Q = random_symmetric_positive_definite_matrix(nx)
    Q = np.stack([Q for k in range(N)], -1)  # copy Q for each time
    R = random_symmetric_positive_definite_matrix(nu)
    system = {'A': A, 'B': B, 'C': C, 'C0': C0, 'H': H, 'D': D,
              'D0': D0, 'E0': E0, 'Q': Q, 'R': R, 'X1': X1, 'S1': S1}
    return system


def random_kalman_lqg_LTV_system(time_samples):
    """ Generate a random time-varying system for the optimal control algorithm
    implemented in Todorov's kalman_lqg.m MATLAB code. """
    N = time_samples    # number of discrete time samples in state trajectory
    
    # number of state variables
    nx = randint(2,5)

    # number of control inputs
    nu = randint(2,5)

    # number of observable outputs
    ny = randint(2,5)

    # number of additive process noise variables
    np0 = randint(2,5)

    # number of control dependent process noise varaibles
    npc = randint(2,5)

    # number of additive measurement noise variables
    nm0 = randint(2,5)

    # number of state dependent measurement noise variables
    nms = randint(2,5)

    # number of internal noise variables
    ni = randint(2,5)

    # scale factor for noise matrices
    noise_scale = 1e-1

    """ generate a random non-linear open loop system
    x(k+1) = A*x(k) + B*u(k) """
    # system dynamics matrix, A
    A = random_nonsingular_matrices(nx, nx, N-1)

    # control input matrix, B
    B = random_nonsingular_matrices(nx, nu, N-1)

    # control input dependent process noise matrices, C
    C = noise_scale*randn(nu, nu, npc, N-1)

    # additive process noise matrix, C0
    C0 = noise_scale*randn(nx, np0, N-1)
    
    # measurement matrix, H 
    H = random_nonsingular_matrices(ny, nx, N)

    # state dependent measurement noise matrices, D
    D = noise_scale*randn(ny, nx, nms, N)

    # additive measurement noise matrix, D0
    D0 = noise_scale*randn(ny, nm0, N)
    
    # internal noise that directly affects the state estimate
    # zero in LQG systems
    #E0 = noise_scale*randn(nx, ni, N-1)
    E0 = np.zeros([nx, ni, N-1])

    # pick a random initial state and initial covariance matrix
    X1 = randn(nx, 1)
    S1 = np.identity(nx) 

    # pick random state and control cost matrices
    Q = random_symmetric_positive_definite_matrices(nx, N)
    R = random_symmetric_positive_definite_matrices(nu, N-1)
    system = {'A': A, 'B': B, 'C': C, 'C0': C0, 'H': H, 'D': D,
              'D0': D0, 'E0': E0, 'Q': Q, 'R': R, 'X1': X1, 'S1': S1}
    return system


def dist(positions):
    """ Emulate the single argument version of MATLAB's dist function which
    returns a matrix of the distances between a set of locations. """
    return np.array([abs(np.array(positions) - i) for i in positions])


def size(A, dimension):
    """ Emulate MATLAB's size function. """
    if is_scalar(A):
        size = 1
    elif len(A.shape) < dimension:
        size = 1
    else:
        size = A.shape[dimension - 1]
    return size


def is_scalar(variable):
    """ Treat variable as a scalar if it is a float or an int. """
    return isinstance(variable, float) or isinstance(variable, int)


def stack_array(A, dimensions, N):
    """ If A is not time varying then stack it to get a copy for each time
    point from 0 to N-1 where time is the last dimension. At each time point A
    should have the specified number of dimensions. """
    if len(A.shape) == dimensions: 
        # A is time invariant
        As = np.stack([A for k in range(N)], -1)
    elif len(A.shape) == dimensions + 1: 
        # A is already time varying
        if A.shape[-1] != N:
            import ipdb; ipdb.set_trace()
        assert A.shape[-1] == N, \
                "stack_array: A does not contain enough time points"
        As = A
    elif len(A.shape) < dimensions: 
        raise ValueError('stack_array: too few dimensions in A')
    elif len(A.shape) > dimensions + 1: 
        raise ValueError('stack_array: too many dimensions in A')
    return As


def stack_matrix(A, N):
    """ If the matrix A is not time varying then stack it to get a copy for
    each time point from 0 to N-1. """
    return stack_array(A, 2, N)


def stack_tensor(A, N):
    """ If the tensor A is not time varying then stack it to get a copy for
    each time point from 0 to N-1. """
    return stack_array(A, 3, N)


def get_time_slice(A, dimensions, time):
    """ Return an array with the specified number of dimensions. This array
    represents a matrix or tensor. If the matrix or tensor is time invariant
    then the number of dimensions specified will match the number of dimensions
    in the array A. If the matrix or tensor is time varying then the number of
    dimensions in A will be one larger than specified and the last dimension
    represents time. """
    A_dimensions = len(A.shape)
    if A_dimensions == dimensions:
        time_slice = A
    elif A_dimensions == dimensions + 1:
        time_slice = A.T[time].T
    elif A_dimensions < dimensions:
        raise ValueError('get_time_slice: too few dimensions')
    elif A_dimensions > dimensions + 1:
        raise ValueError('get_time_slice: too many dimensions')
    return time_slice


def get_matrix(A, time):
    """ Return a two-dimensional array from a two or three-dimensional array
    called A. If the array is 3D then time is the last dimension. """
    return get_time_slice(A, 2, time)


def get_tensor(A, time):
    """ Return a three-dimensional array from a three or four-dimensional array
    called A. If the array is 4D then time is the last dimension. """
    return get_time_slice(A, 3, time)


def noise(C0=0, Cx=0, x=0, Cu=0, u=0, NSim=1):
    """ Produce state and control dependent noise. """
    if is_scalar(C0):
        independent_noise = 0
    else:
        assert len(C0.shape) == 2, "C0 must have 2 dimensions"
        independent_noise = C0.dot(randn(C0.shape[1],NSim))
    if is_scalar(Cx):
        state_dep_noise = 0
    else:
        assert len(Cx.shape) == 3, "Cx must have 3 dimensions"
        nCx = Cx.shape[2]
        state_dep_noise = sum([Cx[:,:,i].dot(x)*randn(NSim)
                               for i in range(nCx)])

    if is_scalar(Cu):
        control_dep_noise = 0
    else:
        assert len(Cu.shape) == 3, "Cu must have 3 dimensions"
        nCu = Cu.shape[2]
        control_dep_noise = sum([Cu[:,:,i].dot(u)*randn(NSim)
                                 for i in range(nCu)])
    noise = independent_noise + state_dep_noise + control_dep_noise
    #if not is_scalar(noise):
        # convert column vector to array slice
        #noise = noise[:,0]
    return noise


def initial_state_trajectory(f, x0, xf, nu, N):
    """ Compute the initial state trajectory for the iLQG algorithm.
    It's a straight line from x0 to xf."""
    # Compute the straight line trajectory from x0 to xf.
    dx = (xf.flatten() - x0.flatten()) / float(N-1)
    x = np.array([x0.flatten() + i*dx for i in range(N)]).T
    return x


def trajectory_cost(system, x, u):
    """ Compute the cost of an optimal control trajectory. """
    cost = 0
    N = x.shape[1]
    if 'Q' in system.keys() and 'R' in system.keys():
        """ This is an LQG system.
        Cost = xQx for k=N + sum of xQx + uRu for k from 0 to N-1 """
        for k in range(N-1):
            Q = get_matrix(system['Q'], k)
            R = get_matrix(system['R'], k)
            cost += x[:,k].dot(Q).dot(x[:,k]) + u[:,k].dot(R).dot(u[:,k])
        cost += h(x[:,-1])
    elif 'l' in system.keys() and 'h' in system.keys():
        """ This is an iLQG system.
        Cost = h(x[T]) + integral of l(x,u,t) for t from 0 to T
        Where T = (N-1) * dt """
        dt = 1.0  # until there's a reason to use something else
        l = system['l']
        h = system['h']
        for k in range(N-1):
            cost += l(x[:,k],u[:,k],k)*dt
        cost += h(x[:,-1])
    else:
        raise KeyError('trajectory_cost: unknown system type')
    return cost


def compute_control(solution, x_hat, k):
    """ Compute the control input given an optimal control law, solution,
    and a state estimate, x_hat. """
    assert len(x_hat.shape) == 2, "x_hat vector(s) must be in columns"
    #import ipdb; ipdb.set_trace()
    if 'x_n' in solution.keys() and 'u_n' in solution.keys():
        # solution is in iLQG format
        L = solution['L']
        l = solution['l']
        x_n = solution['x_n']
        u_n = solution['u_n']
        x_hat_ilqg = x_hat - x_n[:,[k]]
        u = L[:,:,k].dot(x_hat_ilqg) + l[:,[k]] + u_n[:,[k]]
    else:
        # assume solution is in LQG format
        L = solution['L']
        u = -L[:,:,k].dot(x_hat)
    return u


def compute_state_estimate(system, solution, x_hat, u, y, k):
    """ Compute an updated state estimate given a system description, an
    optimal control law, the current state estimate, the control input, and
    a noisy observation. """
    assert len(x_hat.shape) == 2, "x_hat vector(s) must be in columns"
    assert len(u.shape) == 2, "u vector(s) must be in columns"
    assert len(y.shape) == 2, "y vector(s) must be in columns"
    nx = x_hat.shape[0]
    nu = u.shape[0]
    ny = y.shape[0]
    K = solution['K']
    if 'x_n' in solution.keys() and 'u_n' in solution.keys():
        # solution is in iLQG format
        A = get_matrix(solution['system']['A'], k)[0:nx,0:nx]
        B = get_matrix(solution['system']['B'], k)[0:nx,0:nu]
        H = get_matrix(solution['system']['H'], k)[0:ny,0:nx]
        H_lqg = get_matrix(system['H'], k)
        x_n = solution['x_n']
        u_n = solution['u_n']
        x_hat_ilqg = x_hat - x_n[:,[k]]
        u_ilqg = u - u_n[:,[k]]
        y_ilqg = y - H_lqg.dot(x_n[:,[k]])
        new_x_hat = (A.dot(x_hat_ilqg) + B.dot(u_ilqg)
                     + K[:,:,k].dot(y_ilqg - H.dot(x_hat_ilqg)) + x_n[:,[k+1]])
    else:
        # assume solution is in LQG format
        A = get_matrix(system['A'], k)
        B = get_matrix(system['B'], k)
        H = get_matrix(system['H'], k)
        new_x_hat = (A.dot(x_hat) + B.dot(u)
                     + K[:,:,k].dot(y - H.dot(x_hat)))
    return new_x_hat


def compute_cost(system, solution, NSim=1, deterministic=False):
    """
    Use the cost function x.T*Q*x + u.T*R*u to compute the total cost for NSim
    state trajectories starting from x0 and obeying these equations:
    x(k+1) = A(k)*x(k) + B(k)*L(k)*x_hat(k) + process_noise()
    y(k) = H(k)*x(k) + observation_noise()
    x_hat(k+1) = A(k)*x_hat(k) + B(k)*L(k)*x_hat(k) + K(k)*(y(k) - H*x_hat(k))
    """
    x0 = system['X1'].flatten() # make sure x0 is a 1D vector
    A = system['A']
    B = system['B']
    C0 = system['C0']
    C = system['C']
    H = system['H']
    D0 = system['D0']
    D = system['D']
    Q = system['Q']
    R = system['R']
    E0 = system['E0']
    N = Q.shape[2]
    K = solution['K']
    L = solution['L']
    assert len(K.shape) == 3 and K.shape[2] == N-1, \
            "K must contain a matrix for each time step"
    assert len(L.shape) == 3 and L.shape[2] == N-1, \
            "L must contain a matrix for each time step"
    # Make copies of time-invariant matrices for appropriate time points.
    A = stack_matrix(A, N-1)
    B = stack_matrix(B, N-1)
    C0 = stack_matrix(C0, N-1)
    C = stack_tensor(C, N-1)
    H = stack_matrix(H, N)
    D0 = stack_matrix(D0, N)
    D = stack_tensor(D, N)
    R = stack_matrix(R, N-1)
    cost = np.zeros(NSim)
    x = np.array([x0 for i in range(NSim)]).T
    x_hat = x
    if deterministic:
        # no noise
        observation_noise = lambda k: noise()
        control_dep_noise = lambda k: noise()
        process_noise = lambda k: noise()
    else:
        observation_noise = lambda k: noise(x=x, Cx=D[:,:,:,k], C0=D0[:,:,k],
                                            NSim=NSim)
        control_dep_noise = lambda k: noise(u=u, Cu=C[:,:,:,k], NSim=NSim)
        process_noise = lambda k: noise(C0=C0[:,:,k], NSim=NSim)
    for k in range(N-1):
        #print(k)
        #print(x_hat.T)
        #print(np.array([x[:,i].dot(Q[:,:,k]).dot(x[:,i]) for i in
        #                range(NSim)]))
        cost += np.array([x[:,i].dot(Q[:,:,k]).dot(x[:,i]) for i in range(NSim)])
        u = compute_control(solution, x_hat, k)
        u = u + control_dep_noise(k)
        #print(u.T)
        #print(np.array([u[:,i].dot(R[:,:,k]).dot(u[:,i]) for i in
        #                range(NSim)]))
        cost += np.array([u[:,i].dot(R[:,:,k]).dot(u[:,i]) for i in range(NSim)])
        y = H[:,:,k].dot(x) + observation_noise(k)
        x = A[:,:,k].dot(x) + B[:,:,k].dot(u) + process_noise(k)
        x_hat = compute_state_estimate(system, solution, x_hat, u, y, k)
    #print(x_hat.T)

    cost += np.array([x[:,i].dot(Q[:,:,N-1]).dot(x[:,i]) for i in range(NSim)])
    return cost


