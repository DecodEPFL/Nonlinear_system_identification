import torch
import numpy as np
#------------------data collection-----------------------------
#closed loop data with different initial conditions
def generate_closed_loop_data_different_x0(sys, controller, num_signals, horizon, input_dim, output_dim, state_dim):
    # Predefine tensors to store the results
    y_data = torch.zeros((num_signals, horizon, output_dim))  # Position data
    u_data = torch.zeros((num_signals, horizon, input_dim))  # Control input data

    # Run the closed loop for multiple trajectories
    for signal in range(num_signals):

        # Set initial conditions for this trajectory
        x = 1 + 9 * torch.rand(state_dim)
        y = x[0:output_dim]

        # Store initial conditions
        y_data[signal, 0, :] = y

        for t in range(horizon - 1):
            # Calculate the control input (Proportional control)
            control_input = controller.forward(y)
            u = control_input

            # Apply the dynamics model to get next state
            x, y = sys.forward(x, u)
            y += torch.randn_like(y) * 0.01  # noise on output

            # Store the results for this time step
            y_data[signal, t + 1, :] = y
            u_data[signal, t + 1, :] = u

    # Now y_data and u_data will contain the position and control input trajectories for all batches
    return y_data, u_data
#closed loop data with different exciting signals
def generate_closed_loop_data_different_u(sys, controller, num_signals, horizon, input_dim, output_dim, x_0):
    # Predefine tensors to store the results
    y_data = torch.zeros((num_signals, horizon, output_dim))  # Position data
    u_data = torch.zeros((num_signals, horizon, input_dim))  # Control input data

    # Run the closed loop for multiple trajectories
    for signal in range(num_signals):

        # Set initial conditions for this trajectory
        x = x_0
        y = x[0:output_dim]

        # Store initial conditions
        y_data[signal, 0, :] = y

        exciting_amplitude = torch.rand(1).item() /3 # random amplitude
        exciting_frequency = torch.rand(1).item() * 0.4 + 0.1  # random frequency

        for t in range(horizon - 1):
            # Calculate the control input (Proportional control)
            control_input = controller.forward(y)

            # Generate the exciting signal (use float32 for consistency)
            if sys.vehicle:
                exciting_signal = exciting_amplitude * torch.tensor([np.sin(exciting_frequency * t), np.cos(exciting_frequency * t)], dtype=torch.float32)
            else:
                exciting_signal = exciting_amplitude * torch.tensor([np.sin(exciting_frequency * t)], dtype=torch.float32)
            u = control_input + exciting_signal

            # Apply the dynamics model to get next state
            x, y = sys.forward(x, u)
            y += torch.randn_like(y) * 0.01  # noise on output

            # Store the results for this time step
            y_data[signal, t + 1, :] = y
            u_data[signal, t + 1, :] = u

    # Now y_data and u_data will contain the position and control input trajectories for all batches
    return y_data, u_data

#closed loop data with different exciting signals and initial conditions
def generate_closed_loop_data_different_x0_and_u(sys, controller, num_signals, horizon, input_dim, output_dim, state_dim):
    # Predefine tensors to store the results
    y_data = torch.zeros((num_signals, horizon, output_dim))  # Position data
    u_data = torch.zeros((num_signals, horizon, input_dim))  # Control input data

    # Run the closed loop for multiple trajectories
    for signal in range(num_signals):
        # Set initial conditions for this trajectory
        x = 1 + 9 * torch.rand(state_dim)  # Random initial state
        y = x[0:2]
        # Store initial conditions
        y_data[signal, 0, :] = y

        exciting_amplitude = torch.rand(1).item() /3  # random amplitude
        exciting_frequency = torch.rand(1).item() * 0.4 + 0.1  # random frequency

        for t in range(horizon - 1):
            # Calculate the control input (Proportional control)
            control_input = controller.forward(y)

            # Generate the exciting signal (use float32 for consistency)
            if sys.vehicle:
                exciting_signal = exciting_amplitude * torch.tensor(
                    [np.sin(exciting_frequency * t), np.cos(exciting_frequency * t)], dtype=torch.float32)
            else:
                exciting_signal = exciting_amplitude * torch.tensor([np.sin(exciting_frequency * t)], dtype=torch.float32)
            u = control_input + exciting_signal

            # Apply the dynamics model to get next state
            x, y = sys.forward(x, u)
            y += torch.randn_like(y) * 0.01  # noise on output

            # Store the results for this time step
            y_data[signal, t + 1, :] = y
            u_data[signal, t + 1, :] = u
    # Now y_data and u_data will contain the position and control input trajectories for all batches
    return y_data, u_data