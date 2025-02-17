import torch

def set_params():
    # # # # # # # # Parameters # # # # # # # #

    #Model
    mass = 1.0  # Mass of the vehicle (kg)
    ts = 0.05  # Sampling time (s)
    drag_coefficient_1 = 1.  # Drag coefficient 1 (N·s/m)
    drag_coefficient_2 = 0.1  # Drag coefficient 2 (N·s/m)
    initial_position = torch.tensor([2.0, 2.0])  # Initial position (m)
    initial_velocity = torch.tensor([0.0, 0.0])  # Initial velocity (m/s)
    target_position = torch.tensor([0.0, 0.0])  # Target position (m)
    input_dim = 2
    state_dim = 4

    #Controller
    Kp = torch.tensor([[3, 0.0], [0.0, 3]])
    Kd = torch.tensor([[4, 0.0], [0.0, 4]])

    #Dataset
    duration = 5
    num_signals = 20
    num_training = int(num_signals * 4 / 5)
    num_validation = num_signals - num_training

    #PB training
    min_dist = 1.  # min distance for collision avoidance
    t_end = 500
    n_agents = 1
    x0 = torch.tensor([0., 0., 0., 0.])
    xbar = torch.tensor([0., 0., 0., 0.])
    linear = False

    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 1000
    Q = torch.kron(torch.eye(n_agents), torch.diag(torch.tensor([1, 1, 1, 1.])))
    alpha_u = 0.1  # Regularization parameter for penalizing the input
    alpha_ca = 100
    alpha_obst = 5e3
    n_xi = 10  # \xi dimension -- number of states of REN
    l = 10  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
    n_traj = 5  # number of trajectories collected at each step of the learning
    std_ini = 0.2  # standard deviation of initial conditions
    return min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, \
           l, n_traj, std_ini, mass, ts, drag_coefficient_1, drag_coefficient_2, initial_position, initial_velocity, \
           target_position, input_dim, state_dim, Kp, Kd, duration, num_signals, num_training, num_validation