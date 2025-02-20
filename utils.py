import torch

def set_params():
    # # # # # # # # Parameters # # # # # # # #

    #Model
    mass = 1.0  # Mass of the vehicle (kg)
    ts = 0.05  # Sampling time (s)
    drag_coefficient_1 = 1.  # Drag coefficient 1 (N·s/m)
    drag_coefficient_2 = 0.1  # Drag coefficient 2 (N·s/m)
    y_target = torch.tensor([0.0, 0.0])  # Output target: position (m)
    x_0 = torch.tensor([2.0, 2.0, 0., 0.]) # Initial state: position (m) and velocity (m/s)
    input_dim = 2
    state_dim = 4
    output_dim = 2

    #Dataset
    horizon = 100
    num_signals = 20

    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 1

    #Model for system identification
    n_xi = 8  # \xi dimension -- number of states of REN
    l = 8  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN

    return learning_rate, epochs, n_xi, l, mass, ts, drag_coefficient_1, drag_coefficient_2, x_0, y_target, input_dim, state_dim, output_dim, horizon, num_signals

def set_params_2():
    # # # # # # # # Parameters # # # # # # # #

    #Model
    x0 = torch.tensor([5.0])  # Initial state
    input_dim = 1
    state_dim = 1
    output_dim = 1
    input_noise_std = 0.1
    output_noise_std = 0.1


    #Dataset
    horizon = 100
    num_signals = 40
    batch_size = 2
    ts = 0.05  # Sampling time (s)

    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 500

    #Model for system identification
    n_xi = 8  # \xi dimension -- number of states of REN
    l = 8  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN

    return x0, input_dim, state_dim, output_dim, input_noise_std, output_noise_std, horizon, num_signals, batch_size, ts, learning_rate, epochs, n_xi, l