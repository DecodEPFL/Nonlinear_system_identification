import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
from models import PointMassVehicle, PsiU, NonLinearModel, NonLinearController, ProportionalController
from dataset import generate_closed_loop_data_different_x0_and_u, generate_closed_loop_data_different_x0, generate_closed_loop_data_different_u
from utils import set_params
import matplotlib.pyplot as plt
from SSMs import DWN, DWNConfig
import math
from argparse import Namespace

# Define simulation parameters
learning_rate, epochs, n_xi, l, mass, ts, drag_coefficient_1, drag_coefficient_2, x_0, y_target, input_dim, state_dim, output_dim, horizon, num_signals = set_params()
vehicle = False
# Create the model
if vehicle:
    sys = PointMassVehicle(mass, ts, drag_coefficient_1, drag_coefficient_2)
    Kp = torch.tensor([[3, 0.0], [0.0, 3]])
    controller = ProportionalController(Kp, y_target)
else:
    sys = NonLinearModel()
    controller = NonLinearController()
    x_0 = torch.tensor([2.0])  # Initial state: position (m) and velocity (m/s)
    input_dim = 1
    state_dim = 1
    output_dim = 1

y_data, u_data = generate_closed_loop_data_different_x0(sys, controller, num_signals, horizon, input_dim, output_dim, state_dim)
y_data_2, u_data_2 = generate_closed_loop_data_different_u(sys, controller, num_signals, horizon, input_dim, output_dim, x_0)
y_data_3, u_data_3 = generate_closed_loop_data_different_x0_and_u(sys, controller, num_signals, horizon, input_dim, output_dim, state_dim)
if sys.vehicle:
    # CLOSED LOOP FOR 1 TRAJECTORY WITH BASE P CONTROLLER
    # Initial conditions
    x = x_0
    y = x[0:2]

    # Predefine tensors to store the results
    positions_closed = torch.zeros((horizon, 2))  # Store all positions
    velocities_closed = torch.zeros((horizon, 2))  # Store all velocities

    # Set initial conditions
    positions_closed[0] = y
    velocities_closed[0] = x[2:]

    for t in range(horizon - 1):  # Ensure indices align
        control_input = controller.forward(y)
        u = control_input
        x, y = sys.forward(x, u)
        y += torch.randn_like(y) * 0.01  # noise on output
        positions_closed[t + 1] = y  # Store next state
        velocities_closed[t + 1] = x[2:]  # Store next velocity

    # Extract positions for plotting
    x_positions = positions_closed[:, 0].numpy()
    y_positions = positions_closed[:, 1].numpy()

    # Plot the trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(x_positions, y_positions, marker='o', linestyle='-', label='sys Trajectory')
    plt.scatter(x_0[0], x_0[1], color='green', marker='s', label='Start')
    plt.scatter(y_target[0], y_target[1], color='red', marker='x', label='Target')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('sys Trajectory over Time')
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
else:
    # ----------plot one closed loop trajectory--------------------
    # Initial state (random or specific)
    x = torch.randn(state_dim)
    y = x[:output_dim]  # Extract initial output

    # Storage for plotting
    y_traj = torch.zeros((horizon, output_dim))
    u_traj = torch.zeros((horizon, input_dim))

    # Simulate closed-loop system
    for t in range(horizon):
        # Store current state/output
        y_traj[t, :] = y

        # Compute control input
        u = controller.forward(y)
        u_traj[t, :] = u

        # Apply system dynamics
        x, y = sys.forward(x, u)
    time = range(horizon)

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(time, y_traj.numpy(), label=["y1"])
    plt.xlabel("Time Step")
    plt.ylabel("Output (y)")
    plt.title("Closed-loop Output Trajectory")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, u_traj.numpy(), label=["u1"
                                          ], linestyle="dashed")
    plt.xlabel("Time Step")
    plt.ylabel("Control Input (u)")
    plt.title("Control Input Trajectory")
    plt.legend()

    plt.tight_layout()
    plt.show()

if sys.vehicle:
    # Plot the trajectories
    plt.figure(figsize=(10, 6))
    for signal in range(3):
        plt.plot(y_data[signal, :, 0].numpy(), y_data[signal, :, 1].numpy(), label=f'Trajectory {signal+1}')

    plt.title('Position Trajectories starting from different initial conditions')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    # Plot output trajectory

    plt.figure(figsize=(10, 6))

    # Extract trajectory data
    time = range(horizon)

    # Plot output trajectories
    plt.subplot(2, 1, 1)
    for signal in range(3):
        y_traj = y_data[signal].numpy()  # Convert to numpy for plotting
        plt.plot(time, y_traj, label=[f"y{i + 1} (Signal {signal})" for i in range(output_dim)])

    plt.xlabel("Time Step")
    plt.ylabel("Output (y)")
    plt.title("Closed-loop Output Trajectories")
    plt.legend()

    # Plot control input trajectories
    plt.subplot(2, 1, 2)
    for signal in range(3):
        u_traj = u_data[signal].numpy()
        plt.plot(time, u_traj, linestyle="dashed", label=[f"u{i + 1} (Signal {signal})" for i in range(input_dim)])

    plt.xlabel("Time Step")
    plt.ylabel("Control Input (u)")
    plt.title("Control Input Trajectories")
    plt.legend()

    plt.tight_layout()
    plt.show()

if sys.vehicle:
    # Plot the trajectories
    plt.figure(figsize=(10, 6))

    for signal in range(3):
        plt.plot(y_data_2[signal, :, 0].numpy(), y_data_2[signal, :, 1].numpy(), label=f'Trajectory {signal+1}')

    plt.title('Position Trajectories corresponding to different exciting signals')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    # Plot output trajectory
    plt.figure(figsize=(10, 6))

    # Extract trajectory data
    time = range(horizon)

    # Plot output trajectories
    plt.subplot(2, 1, 1)
    for signal in range(3):
        y_traj = y_data_2[signal].numpy()  # Convert to numpy for plotting
        plt.plot(time, y_traj, label=[f"y{i + 1} (Signal {signal})" for i in range(output_dim)])

    plt.xlabel("Time Step")
    plt.ylabel("Output (y)")
    plt.title("Closed-loop Output Trajectories")
    plt.legend()

    # Plot control input trajectories
    plt.subplot(2, 1, 2)
    for signal in range(3):
        u_traj = u_data_2[signal].numpy()
        plt.plot(time, u_traj, linestyle="dashed", label=[f"u{i + 1} (Signal {signal})" for i in range(input_dim)])

    plt.xlabel("Time Step")
    plt.ylabel("Control Input (u)")
    plt.title("Control Input Trajectories")
    plt.legend()

    plt.tight_layout()
    plt.show()

if sys.vehicle:
    # Plot the trajectories
    plt.figure(figsize=(10, 6))

    for signal in range(3):
        plt.plot(y_data_3[signal, :, 0].numpy(), y_data_3[signal, :, 1].numpy(), label=f'Trajectory {signal+1}')

    plt.title('Position Trajectories corresponding to different exciting signals and initial conditions')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    # Plot output trajectory
    plt.figure(figsize=(10, 6))

    # Extract trajectory data
    time = range(horizon)

    # Plot output trajectories
    plt.subplot(2, 1, 1)
    for signal in range(3):
        y_traj = y_data_3[signal].numpy()  # Convert to numpy for plotting
        plt.plot(time, y_traj, label=[f"y{i + 1} (Signal {signal})" for i in range(output_dim)])

    plt.xlabel("Time Step")
    plt.ylabel("Output (y)")
    plt.title("Closed-loop Output Trajectories")
    plt.legend()

    # Plot control input trajectories
    plt.subplot(2, 1, 2)
    for signal in range(3):
        u_traj = u_data_3[signal].numpy()
        plt.plot(time, u_traj, linestyle="dashed", label=[f"u{i + 1} (Signal {signal})" for i in range(input_dim)])

    plt.xlabel("Time Step")
    plt.ylabel("Control Input (u)")
    plt.title("Control Input Trajectories")
    plt.legend()

    plt.tight_layout()
    plt.show()

#----------define models--------------
# ssms
cfg = {
    "n_u": input_dim,
    "n_y": output_dim,
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

# Define the loss function
MSE = nn.MSELoss()

#dataset
#select data 1 for different initial conditions or data 2 for different exciting signals 3 for both
input_data_training = u_data
output_data_training = y_data
y_hat_train = torch.zeros(output_data_training.shape)
time_plot = np.arange(0, input_data_training.shape[1] * ts, ts)

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
    y_hat_train, _ = Qg_SSM(input_data_training, state=None, mode="scan")

    # Calculate the mean squared error loss
    loss = MSE(y_hat_train, output_data_training)
    loss.backward()  # Backpropagate to compute gradients

    # Update model parameters
    optimizer.step()

    # Print loss for each epoch
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss

if sys.vehicle:
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
        plt.title(f"Real vs Modelled Outputs with SSMs for Signal {i} in training set")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()

    plt.show()
else:
    plt.figure(figsize=(12, 8))

    # Plot for each selected signal
    for i in range(2):
        plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(), label="Real Output X",
                 color="blue")
        plt.plot(time_plot, y_hat_train[i, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X",
                 linestyle="--", color="orange")
        plt.title(f"Real vs Modelled Outputs with SSMs for Signal {i} in training set")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()

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
                u_K = torch.zeros(input_dim)
                state = None
            u_ext = input_data_training[n, t, :]
            u = u_ext - u_K
            u = u.view(1, 1, input_dim)  # Reshape input
            y_hat, state = Qg_SSM(u, state=state, mode="loop")
            y_hat = y_hat.squeeze(0).squeeze(0)
            u_K = controller.forward(y_hat)
            loss = loss + MSE(output_data_training[n, t, :], y_hat[:])
            y_hat_train[n, t, :] = y_hat.detach()

    # Normalize training loss
    loss /= (input_data_training.shape[0] * input_data_training.shape[1])
    loss.backward()
    optimizer.step()

    # Print training loss for this epoch
    print(f"Epoch: {epoch + 1} \t||\t Training Loss: {loss}")
    LOSS[epoch] = loss.item()

if sys.vehicle:
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
        plt.title(f"Real vs Modelled Outputs with SSMs for Signal {i} in training set")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()

    plt.show()
else:
    plt.figure(figsize=(12, 8))

    # Plot for each selected signal
    for i in range(2):
        plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(), label="Real Output X",
                 color="blue")
        plt.plot(time_plot, y_hat_train[i, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X",
                 linestyle="--", color="orange")
        plt.title(f"Real vs Modelled Outputs with SSMs for Signal {i} in training set")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()

        plt.show()
