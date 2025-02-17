import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
from models import PointMassVehicle, PsiU
from utils import set_params
import matplotlib.pyplot as plt
from SSMs import DWN, DWNConfig
import math
from argparse import Namespace

# Define simulation parameters
# Parameters
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, l, \
    n_traj, std_ini, mass, ts, drag_coefficient_1, drag_coefficient_2, initial_position, initial_velocity, \
    target_position, input_dim, state_dim, Kp, Kd, duration, num_signals, num_training, num_validation = set_params()

horizon = int(duration/ts)
# Create the vehicle model
vehicle = PointMassVehicle(mass, ts, drag_coefficient_1, drag_coefficient_2)

#CLOSED LOOP FOR 1 TRAJECTORY WITH BASE P CONTROLLER
# Initial conditions
p = initial_position
q = initial_velocity

# Predefine tensors to store the results
positions_closed = torch.zeros((horizon, 2))  # Store all positions
velocities_closed = torch.zeros((horizon, 2))  # Store all velocities

# Set initial conditions
positions_closed[0] = p
velocities_closed[0] = q

for t in range(horizon - 1):  # Ensure indices align
    control_input = torch.matmul(Kp, target_position - p)
    F = control_input
    p, q = vehicle.forward(p, q, F)
    p += torch.randn_like(p) * 0.01 #noise on output
    positions_closed[t + 1] = p  # Store next state
    velocities_closed[t + 1] = q  # Store next velocity

# Extract positions for plotting
x_positions = positions_closed[:, 0].numpy()
y_positions = positions_closed[:, 1].numpy()

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_positions, y_positions, marker='o', linestyle='-', label='Vehicle Trajectory')
plt.scatter(initial_position[0], initial_position[1], color='green', marker='s', label='Start')
plt.scatter(target_position[0], target_position[1], color='red', marker='x', label='Target')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Trajectory over Time')
plt.legend()
plt.grid()
plt.show()


# Plot position trajectory
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(range(horizon), positions_closed[:, 0], label='Position X')
plt.plot(range(horizon), positions_closed[:, 1], label='Position Y')
plt.title('Position Trajectory')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()

# Plot velocity trajectory
plt.subplot(2, 1, 2)
plt.plot(range(horizon), velocities_closed[:, 0], label='Velocity X')
plt.plot(range(horizon), velocities_closed[:, 1], label='Velocity Y')
plt.title('Velocity Trajectory')
plt.xlabel('Time Step')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()


# Parameters
num_batches = num_training  # Number of trajectories
output_dim = int(state_dim/2)  # Assuming 2D position (x, y)
input_dimension = input_dim  # Control input dimension

#closed loop data with different initial conditions
# Predefine tensors to store the results
y_data = torch.zeros((num_batches, horizon, output_dim))  # Position data
u_data = torch.zeros((num_batches, horizon, input_dimension))  # Control input data

# Run the closed loop for multiple trajectories
for batch in range(num_batches):

    # Set initial conditions for this trajectory
    p = torch.tensor([torch.randn(1).item(), torch.randn(1).item()])  # Random 2D position
    q = torch.tensor([torch.randn(1).item(), torch.randn(1).item()])  # Random 2D velocity

    # Store initial conditions
    y_data[batch, 0, :] = p

    for t in range(horizon - 1):
        # Calculate the control input (Proportional control)
        control_input = torch.matmul(Kp, target_position - p)
        F = control_input

        # Apply the dynamics model to get next state
        p, q = vehicle.forward(p, q, F)
        p += torch.randn_like(p) * 0.01  # noise on output

        # Store the results for this time step
        y_data[batch, t + 1, :] = p
        u_data[batch, t + 1, :] = F

# Now y_data and u_data will contain the position and control input trajectories for all batches

# Plot the trajectories
plt.figure(figsize=(10, 6))

for batch in range(3):
    plt.plot(y_data[batch, :, 0].numpy(), y_data[batch, :, 1].numpy(), label=f'Trajectory {batch+1}')

plt.title('Position Trajectories')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.legend()
plt.grid(True)
plt.show()

#closed loop data with different exciting signals
# Predefine tensors to store the results
y_data_2 = torch.zeros((num_batches, horizon, output_dim))  # Position data
u_data_2 = torch.zeros((num_batches, horizon, input_dimension))  # Control input data

# Run the closed loop for multiple trajectories
for batch in range(num_batches):
    # Set initial conditions for this trajectory
    p = initial_position
    q = initial_velocity

    # Store initial conditions
    y_data_2[batch, 0, :] = p

    exciting_amplitude = torch.rand(1).item() * 10  # random amplitude
    exciting_frequency = torch.randint(1, 501, (1,)).item()  # random frequency

    for t in range(horizon - 1):
        # Calculate the control input (Proportional control)
        control_input = torch.matmul(Kp, target_position - p)

        # Generate the exciting signal (use float32 for consistency)
        exciting_signal = exciting_amplitude * torch.tensor([np.sin(exciting_frequency * t), np.cos(exciting_frequency * t)], dtype=torch.float32)
        F = control_input + exciting_signal

        # Apply the dynamics model to get next state
        p, q = vehicle.forward(p, q, F)
        p += torch.randn_like(p) * 0.01  # noise on output

        # Store the results for this time step
        y_data_2[batch, t + 1, :] = p
        u_data_2[batch, t + 1, :] = F

# Now y_data and u_data will contain the position and control input trajectories for all batches

# Plot the trajectories
plt.figure(figsize=(10, 6))

for batch in range(3):
    plt.plot(y_data_2[batch, :, 0].numpy(), y_data_2[batch, :, 1].numpy(), label=f'Trajectory {batch+1}')

plt.title('Position Trajectories')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.legend()
plt.grid(True)
plt.show()

#closed loop data with different exciting signals and initial conditions
# Predefine tensors to store the results
y_data_3 = torch.zeros((num_batches, horizon, output_dim))  # Position data
u_data_3 = torch.zeros((num_batches, horizon, input_dimension))  # Control input data

# Run the closed loop for multiple trajectories
for batch in range(num_batches):
    # Set initial conditions for this trajectory
    p = torch.tensor([torch.randn(1).item(), torch.randn(1).item()])  # Random 2D position
    q = torch.tensor([torch.randn(1).item(), torch.randn(1).item()])  # Random 2D velocity

    # Store initial conditions
    y_data_3[batch, 0, :] = p

    exciting_amplitude = torch.rand(1).item() * 10  # random amplitude
    exciting_frequency = torch.randint(1, 501, (1,)).item()  # random frequency

    for t in range(horizon - 1):
        # Calculate the control input (Proportional control)
        control_input = torch.matmul(Kp, target_position - p)

        # Generate the exciting signal (use float32 for consistency)
        exciting_signal = exciting_amplitude * torch.tensor([np.sin(exciting_frequency * t), np.cos(exciting_frequency * t)], dtype=torch.float32)
        F = control_input + exciting_signal

        # Apply the dynamics model to get next state
        p, q = vehicle.forward(p, q, F)
        p += torch.randn_like(p) * 0.01  # noise on output

        # Store the results for this time step
        y_data_3[batch, t + 1, :] = p
        u_data_3[batch, t + 1, :] = F

# Now y_data and u_data will contain the position and control input trajectories for all batches

# Plot the trajectories
plt.figure(figsize=(10, 6))

for batch in range(3):
    plt.plot(y_data_3[batch, :, 0].numpy(), y_data_3[batch, :, 1].numpy(), label=f'Trajectory {batch+1}')

plt.title('Position Trajectories')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.legend()
plt.grid(True)
plt.show()

#----------define models--------------
# ssms
cfg = {
    "n_u": 2,
    "n_y": 2,
    "d_model": 5,
    "d_state": 5,
    "n_layers": 3,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi,
    "r_min": 0.7,
    "r_max": 0.98,
    "gamma": False,
    "trainable": False,
    "gain": 2.4
}
cfg = Namespace(**cfg)

# Build model
config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, gamma=cfg.gamma, trainable=cfg.trainable, gain=cfg.gain)
Qg_SSM = DWN(cfg.n_u, cfg.n_y, config)

#create the model Qg REN
Qg_REN = PsiU(input_dim, output_dim, n_xi, l)

# Define the loss function
MSE = nn.MSELoss()

#dataset
#select data 1 for different initial conditions or data 2 for different exciting signals 3 for both
input_data_training = u_data_2
output_data_training = y_data_2
y_hat_train = torch.zeros(output_data_training.shape)

#-----------------------------closedloop sysid training of G directly through SSM------------------------

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(Qg_SSM.parameters(), lr=learning_rate)
optimizer.zero_grad()

# Training loop settings
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    # Adjust learning rate at specific epochs
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(Qg_SSM.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(Qg_SSM.parameters(), lr=learning_rate)

    optimizer.zero_grad()  # Reset gradients
    loss = 0  # Initialize loss

    # Forward pass through the SSM
    ySSM, _ = Qg_SSM(input_data_training, state=None, mode="scan")
    ySSM = torch.squeeze(ySSM)  # Remove unnecessary dimensions

    # Calculate the mean squared error loss
    loss = MSE(ySSM, output_data_training)
    loss.backward()  # Backpropagate to compute gradients

    # Update model parameters
    optimizer.step()

    # Print loss for each epoch
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss

time_plot = np.arange(0, input_data_training.shape[1] * ts, ts)

for idx in range(2):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_plot, ySSM[idx, 0:len(time_plot), 0].detach().numpy(), label='SSM')
    plt.plot(time_plot, output_data_training[idx, 0:len(time_plot), 0].detach().numpy(), label='y train')
    plt.title("Output Train Single SSM")
    plt.legend()
    plt.show()

#-----------------------------closedloop sysid of S through SSMs------------------------

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(Qg_SSM.parameters(), lr=learning_rate)
optimizer.zero_grad()

# Training loop settings
LOSS = np.zeros(epochs)

validation_losses = []

for epoch in range(epochs):
    # Adjust learning rate at specific epochs
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(Qg_SSM.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(Qg_SSM.parameters(), lr=learning_rate)

    optimizer.zero_grad()
    loss = 0.0

    # Training loop
    for n in range(input_data_training.shape[0]):
        for t in range(input_data_training.shape[1]):
            if t == 0:
                u_K = torch.zeros(2)
                state = None
            u_ext = input_data_training[n, t, :]
            u = u_ext - u_K
            u = u.view(1, 1, 2)  # Reshape input
            y_hat, state = Qg_SSM(u, state=state, mode="loop")
            y_hat = y_hat.squeeze(0).squeeze(0)
            u_K = torch.matmul(Kp, target_position - y_hat)
            loss = loss + MSE(output_data_training[n, t, :], y_hat[:])
            y_hat_train[n, t, :] = y_hat.detach()

    # Normalize training loss
    loss /= (input_data_training.shape[0] * input_data_training.shape[1])
    loss.backward()
    optimizer.step()

    # Print training loss for this epoch
    print(f"Epoch: {epoch + 1} \t||\t Training Loss: {loss}")
    LOSS[epoch] = loss.item()


plt.figure(figsize=(12, 8))

# Plot for each selected signal
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(), label="Real Output X", color="blue")
    plt.plot(time_plot, y_hat_train[i, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X", linestyle="--", color="orange")
    plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 1].detach().numpy(), label="Real Output Y", color="green")
    plt.plot(time_plot, y_hat_train[i, 0:len(time_plot), 1].detach().numpy(), label="Modelled Output Y", linestyle="--", color="red")
    plt.title(f"Real vs Modelled Outputs for Signal {i} in training set")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()

plt.show()

#-----------------------------closedloop sysid of S through RENs------------------------

optimizer = torch.optim.Adam(Qg_REN.parameters(), lr=learning_rate)
optimizer.zero_grad()

# Training loop settings
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    optimizer.zero_grad()  # Reset the gradients before backpropagation
    loss = 0.0  # Initialize loss for this epoch

    # Training loop
    for n in range(input_data_training.shape[0]):  # Iterate over each batch
        xi_ = torch.zeros(Qg_REN.n_xi)  # Reset xi_ for each trajectory
        u_K = torch.zeros(2)  # Reset the control input for each trajectory

        for t in range(input_data_training.shape[1]):  # Iterate over each time step
            u_ext = input_data_training[n, t, :]  # Extract external input
            u = u_ext - u_K  # Apply control input adjustment

            # Get model output
            y_hat, xi_ = Qg_REN.forward(t, u, xi_)

            # Compute the control input
            u_K = torch.matmul(Kp, target_position - y_hat)  # Update control law

            # Accumulate loss across all time steps and samples
            loss += MSE(output_data_training[n, t, :], y_hat)  # Ensure correct loss calculation

            # Store the predicted output (without detaching to allow gradient flow)
            y_hat_train[n, t, :] = y_hat  # Do not detach here to maintain the graph

    # Normalize training loss by batch size and time steps
    loss /= (input_data_training.shape[0] * input_data_training.shape[1])

    # Backpropagate the loss
    loss.backward()

    # Update the model parameters
    optimizer.step()

    # Update the model parameters if needed
    Qg_REN.set_model_param()

    # Print training loss for this epoch
    print(f"Epoch: {epoch + 1} \t||\t Training Loss: {loss.item()}")
    LOSS[epoch] = loss.item()

    plt.figure(figsize=(12, 8))

    # Plot for each selected signal
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(), label="Real Output X",
                 color="blue")
        plt.plot(time_plot, y_hat_train[i, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X",
                 linestyle="--", color="orange")
        plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 1].detach().numpy(), label="Real Output Y",
                 color="green")
        plt.plot(time_plot, y_hat_train[i, 0:len(time_plot), 1].detach().numpy(), label="Modelled Output Y",
                 linestyle="--", color="red")
        plt.title(f"Real vs Modelled Outputs for Signal {i} in training set")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()

    plt.show()