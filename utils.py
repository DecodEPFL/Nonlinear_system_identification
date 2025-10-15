import torch

def set_params():
    # # # # # # # # Parameters # # # # # # # #

    seed = 4
    #Model
    x0 = torch.tensor([5.0])  # Initial state
    input_dim = 1
    state_dim = 1
    output_dim = 1
    input_noise_std = 0.5
    output_noise_std = 0.1


    #Dataset
    horizon = 100
    num_signals = 140
    # Compute split sizes
    train_size = 30
    val_size = 10
    test_size = 100

    batch_size = 2
    ts = 0.05  # Sampling time (s)

    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 500

    #Model for system identification
    n_xi = 8  # \xi dimension -- number of states of REN
    l = 8  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN

    return seed, x0, input_dim, state_dim, output_dim, input_noise_std, output_noise_std, horizon, num_signals, train_size, val_size, test_size, batch_size, ts, learning_rate, epochs, n_xi, l

def set_params_robot():
    # # # # # # # # Parameters # # # # # # # #
    seed = 1

    #Model
    x0 = torch.tensor([20., 10., 10., 0.])  # Initial state
    input_dim = 2
    state_dim = 4
    output_dim = 4
    input_noise_std = 10
    output_noise_std = 0.1
    n_agents = 1


    #Dataset
    horizon = 100
    num_signals = 140
    # Compute split sizes
    train_size = 30
    val_size = 10
    test_size = 100

    batch_size = 2
    ts = 0.05  # Sampling time (s)

    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 500

    #Model for system identification
    n_xi = 8  # \xi dimension -- number of states of REN
    l = 8  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN

    return seed, x0, input_dim, state_dim, output_dim, input_noise_std, output_noise_std, n_agents, horizon, num_signals, train_size, val_size, test_size, batch_size, ts, learning_rate, epochs, n_xi, l

