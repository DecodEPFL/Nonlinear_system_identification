import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
from models import PointMassVehicle, PsiU, NonLinearController, ProportionalController, NonLinearModel
from utils import set_params
from dataset import generate_closed_loop_data_different_x0_and_u, generate_closed_loop_data_different_x0, generate_closed_loop_data_different_u

# Define simulation parameters
learning_rate, epochs, n_xi, l, mass, ts, drag_coefficient_1, drag_coefficient_2, x_0, y_target, input_dim, state_dim, output_dim, horizon, num_signals = set_params()
vehicle = False
plot_dataset = True
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





# Run open-loop simulation
# Simulation parameters

# Generate sinusoidal input
t = torch.arange(horizon, dtype=torch.float32)
u = torch.sin(2 * torch.pi * t / horizon)

x = x_0
x_traj = torch.zeros(horizon + 1)
y_traj = torch.zeros(horizon)
x_traj[0] = x_0

# Run open-loop simulation
for i in range(horizon):
    x, y = sys.forward(x, u[i])
    x_traj[i + 1] = x
    y_traj[i] = y

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t.numpy(), u.numpy(), label='Input (u)')
plt.xlabel('Time step')
plt.ylabel('Input')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t.numpy(), y_traj.numpy(), label='Output (y)', color='r')
plt.xlabel('Time step')
plt.ylabel('Output')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()








y_data, u_data = generate_closed_loop_data_different_x0(sys, controller, num_signals, horizon, input_dim, output_dim, state_dim)
y_data_2, u_data_2 = generate_closed_loop_data_different_u(sys, controller, num_signals, horizon, input_dim, output_dim, x_0)
y_data_3, u_data_3 = generate_closed_loop_data_different_x0_and_u(sys, controller, num_signals, horizon, input_dim, output_dim, state_dim)

#----------define model--------------
#create the model Qg REN
Qg_REN = PsiU(input_dim, output_dim, n_xi, l)

# Define the loss function
MSE = nn.MSELoss()

#dataset
#select data 1 for different initial conditions or data 2 for different exciting signals 3 for both
input_data_training = u_data_2
output_data_training = y_data_2
time_plot = np.arange(0, input_data_training.shape[1] * ts, ts)
time_plot_u = np.arange(0, input_data_training.shape[1] * ts, ts)[1:]

#-----------------------------closedloop sysid training of G directly through RENs------------------------
y_hat_train_G = torch.zeros(output_data_training.shape)

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

        for t in range(input_data_training.shape[1]):  # Iterate over each time step
            u_ext = input_data_training[n, t, :]  # Extract external input
            u = u_ext

            # Get model output
            y_hat, xi_ = Qg_REN.forward(t, u, xi_)

            # Accumulate loss across all time steps and samples
            loss += MSE(output_data_training[n, t, :], y_hat)  # Ensure correct loss calculation

            # Store the predicted output (without detaching to allow gradient flow)
            y_hat_train_G[n, t, :] = y_hat  # Do not detach here to maintain the graph

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

#-----------------------------closedloop sysid of S through RENs------------------------
y_hat_train_S = torch.zeros(output_data_training.shape)
optimizer = torch.optim.Adam(Qg_REN.parameters(), lr=learning_rate)
optimizer.zero_grad()
u_S = torch.zeros(input_data_training.shape)
# Training loop settings
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    optimizer.zero_grad()  # Reset the gradients before backpropagation
    loss = 0.0  # Initialize loss for this epoch

    # Training loop
    for n in range(input_data_training.shape[0]):  # Iterate over each batch
        xi_ = torch.zeros(Qg_REN.n_xi)  # Reset xi_ for each trajectory
        u_K = torch.zeros(input_dim)  # Reset the control input for each trajectory

        for t in range(input_data_training.shape[1]):  # Iterate over each time step
            u_ext = input_data_training[n, t, :]  # Extract external input
            u = u_ext - u_K  # Apply control input adjustment
            # Get model output
            y_hat, xi_ = Qg_REN.forward(t, u, xi_)

            # Compute the control input
            u_K = controller.forward(y_hat)  # Update control law

            # Accumulate loss across all time steps and samples
            loss += MSE(output_data_training[n, t, :], y_hat)

            # Store the predicted output
            if epoch == epochs - 1:
                y_hat_train_S[n, t, :] = y_hat.detach()
                u_S[n, t, :] = u.detach()

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

# --------------------------PLOTS-----------------------------------
if sys.vehicle:
    if plot_dataset:

        # --------------------Plot first dataset-----------------------------------
        plt.figure(figsize=(10, 6))
        for signal in range(3):
            plt.plot(y_data[signal, :, 0].numpy(), y_data[signal, :, 1].numpy(), label=f'Trajectory {signal + 1}')

        plt.title('Position Trajectories starting from different initial conditions')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --------------------Plot second dataset-----------------------------------
        plt.figure(figsize=(10, 6))

        for signal in range(3):
            plt.plot(y_data_2[signal, :, 0].numpy(), y_data_2[signal, :, 1].numpy(), label=f'Trajectory {signal + 1}')

        plt.title('Position Trajectories corresponding to different exciting signals')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --------------------Plot third dataset-----------------------------------
        plt.figure(figsize=(10, 6))

        for signal in range(3):
            plt.plot(y_data_3[signal, :, 0].numpy(), y_data_3[signal, :, 1].numpy(), label=f'Trajectory {signal + 1}')

        plt.title('Position Trajectories corresponding to different exciting signals and initial conditions')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    # --------------Plot identification results for G-----------------
    plt.figure(figsize=(12, 8))

    # Plot for each selected signal
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(), label="Real Output X",
                 color="blue")
        plt.plot(time_plot, y_hat_train_G[i, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X",
                 linestyle="--", color="orange")
        plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 1].detach().numpy(), label="Real Output Y",
                 color="green")
        plt.plot(time_plot, y_hat_train_G[i, 0:len(time_plot), 1].detach().numpy(), label="Modelled Output Y",
                 linestyle="--", color="red")
        plt.title(f"Real vs Modelled Outputs with a REN model for G for Signal {i} in training set")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()

    plt.show()

    # --------------Plot identification results for S-----------------
    plt.figure(figsize=(12, 8))

    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(), label="Real Output X",
                 color="blue")
        plt.plot(time_plot, y_hat_train_S[i, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X",
                 linestyle="--", color="orange")
        plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 1].detach().numpy(), label="Real Output Y",
                 color="green")
        plt.plot(time_plot, y_hat_train_S[i, 0:len(time_plot), 1].detach().numpy(), label="Modelled Output Y",
                 linestyle="--", color="red")
        plt.title(f"Real vs Modelled Outputs with a REN model for S for Signal {i} in training set")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()

    plt.show()

else:
    if plot_dataset:
        # --------------------Plot first dataset-----------------------------------
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
        plt.title("Closed-loop Output Trajectories with different initial conditions")
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

        # --------------------Plot second dataset-----------------------------------
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
        plt.title("Closed-loop Output Trajectories with different external exciting signals")
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

        # --------------------Plot third dataset-----------------------------------
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
        plt.title("Closed-loop Output Trajectories with different initial conditions and external exciting signals")
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

    # --------------Plot identification results for G-----------------

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)

    # Plot for each selected signal in a separate subplot
    for i in range(2):
        axes[i].plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(),
                     label="Real Output X", color="blue")

        axes[i].plot(time_plot, y_hat_train_G[i, 0:len(time_plot), 0].detach().numpy(),
                     label="Modelled Output X", linestyle="--", color="orange")

        axes[i].set_title(f"Real vs Modelled Outputs with a REN model for G - Signal {i}")
        axes[i].set_ylabel("Output")
        axes[i].legend()

    axes[-1].set_xlabel("Time (s)")  # Only set x-label on the last subplot for clarity
    plt.tight_layout()

    # Plot control input u_S in the third subplot
    axes[2].plot(time_plot_u, input_data_training[0, 1:len(time_plot), 0].detach().numpy(),
                 label="Control Input u_G", color="green")

    axes[2].set_title("Control Input u_G")
    axes[2].set_ylabel("Input Value")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # --------------Plot identification results for S-----------------
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)

    # Plot for each selected signal in a separate subplot
    for i in range(2):
        axes[i].plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(),
                     label="Real Output", color="blue")

        axes[i].plot(time_plot, y_hat_train_S[i, 0:len(time_plot), 0].detach().numpy(),
                     label="Modelled Output", linestyle="--", color="orange")

        axes[i].set_title(f"Real vs Modelled Outputs with a REN model for S - Signal {i}")
        axes[i].set_ylabel("Output")
        axes[i].legend()

    axes[-1].set_xlabel("Time (s)")  # Only set x-label on the last subplot for clarity
    plt.tight_layout()

    # Plot control input u_S in the third subplot
    axes[2].plot(time_plot_u, u_S[0, 1:len(time_plot), 0].detach().numpy(),
                 label="Control Input u_S", color="green")

    axes[2].set_title("Control Input u_S")
    axes[2].set_ylabel("Input Value")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# Run open-loop simulation of the identified G
# Simulation parameters

# Generate sinusoidal input
t = torch.arange(horizon, dtype=torch.float32)

xi_ = torch.zeros(Qg_REN.n_xi)
y_traj = torch.zeros(horizon)

# Run open-loop simulation

for i in range(horizon-1):  # Iterate over each time step
    u_ext = input_data_training[0, i, :]  # Extract external input
    u = u_ext
    y_hat, xi_ = Qg_REN.forward(t, u, xi_)
    y_traj[i+1] = y_hat


# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t.numpy(), input_data_training[0, :, 0].detach().numpy(), label='Input (u)')
plt.xlabel('Time step')
plt.ylabel('Input')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t.numpy(), y_traj.detach().numpy(), label='Output (y)', color='r')
plt.xlabel('Time step')
plt.ylabel('Output')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()