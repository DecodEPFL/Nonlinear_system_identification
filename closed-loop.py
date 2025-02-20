import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
from models2 import NonLinearModel, NonLinearController, ClosedLoopSystem
from contractive_REN import ContractiveREN
from utils import set_params_2
from dataset2 import SystemIdentificationDataset
from torch.utils.data import DataLoader, random_split

# Define simulation parameters
x0, input_dim, state_dim, output_dim, input_noise_std, output_noise_std, horizon, num_signals, batch_size, ts, learning_rate, epochs, n_xi, l = set_params_2()

#-------------------------1. Create the plant and controller------------------------------------
sys = NonLinearModel(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim, output_noise_std=output_noise_std)
controller = NonLinearController(input_K_dim=output_dim, output_K_dim=input_dim)
closed_loop = ClosedLoopSystem(sys, controller)

#-------------------------plot open loop data----------------------------------------------------
u_OL = torch.randn((num_signals, horizon, input_dim)) * input_noise_std
y_OL = sys(x0,u_OL, horizon)

# Convert tensors to numpy for plotting
u_OL = u_OL.detach().numpy()
y_OL = y_OL.detach().numpy()

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot Plant Inputs (Controller Outputs)
for i in range(input_dim):
    axes[0].plot(u_OL[:, :, i].T, alpha=0.6)
axes[0].set_title("Plant Input Trajectories (Control Inputs)")
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel("Control Input")

# Plot Plant Outputs
for i in range(output_dim):
    axes[1].plot(y_OL[:, :, i].T, alpha=0.6)
axes[1].set_title("System Output Trajectories")
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("System Output")

plt.tight_layout()
plt.show()

#-------------------------2. Generate closed loop data---------------------------------------------
dataset = SystemIdentificationDataset(num_signals = num_signals, horizon = horizon, input_dim = input_dim, state_dim = state_dim, output_dim = output_dim, closed_loop = closed_loop, input_noise_std = input_noise_std, fixed_x0 = x0)
# Compute split sizes
train_size = int(num_signals/2)
val_size = int(num_signals/4)
test_size = int(num_signals/4)

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


external_input_data = dataset.external_input_data
plant_input_data = dataset.plant_input_data
output_data = dataset.output_data

#-------------------------plot closed loop data---------------------------------------------------

# Convert tensors to numpy for plotting
external_input_np = external_input_data.detach().numpy()
plant_input_np = plant_input_data.detach().numpy()
output_np = output_data.detach().numpy()

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Plot External Inputs
for i in range(input_dim):
    axes[0].plot(external_input_np[:, :, i].T, alpha=0.6)
axes[0].set_title("External Input Trajectories")
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel("External Input")

# Plot Plant Inputs (Controller Outputs)
for i in range(input_dim):
    axes[1].plot(plant_input_np[:, :, i].T, alpha=0.6)
axes[1].set_title("Plant Input Trajectories (Control Inputs)")
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("Control Input")

# Plot System Outputs
for i in range(output_dim):
    axes[2].plot(output_np[:, :, i].T, alpha=0.6)
axes[2].set_title("System Output Trajectories")
axes[2].set_xlabel("Time Step")
axes[2].set_ylabel("System Output")

plt.tight_layout()
plt.show()

#--------------------------3. Define model for sysid---------------------------------------------
#create the model Qg REN
y_init = x0
REN = ContractiveREN(dim_in= input_dim, dim_out= output_dim, dim_internal=n_xi, dim_nl= l, y_init = y_init)

#--------------------------4. Define the loss function and optimizer---------------------------------------------
MSE = nn.MSELoss()

optimizer = torch.optim.Adam(REN.parameters(), lr=learning_rate)
optimizer.zero_grad()
#--------------------------5. Training---------------------------------------------------------------------
input_data_training = plant_input_data
output_data_training = output_data
y_hat_train_G = torch.zeros(output_data_training.shape)
# Training loop settings
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    optimizer.zero_grad()  # Reset the gradients before backpropagation
    loss = 0.0  # Initialize loss for this epoch

    # Training loop
    for n in range(input_data_training.shape[0]):  # Iterate over each batch
        REN.reset()  # Reset xi_ for each trajectory

        for t in range(input_data_training.shape[1]):  # Iterate over each time step
            u_ext = input_data_training[n, t, :].view(1, 1, -1)
            u = u_ext

            # Get model output
            y_hat = REN.forward(u)

            # Accumulate loss across all time steps and samples
            loss += MSE(output_data_training[n, t, :].view(1, 1, -1), y_hat)  # Ensure correct loss calculation

            # Store the predicted output (without detaching to allow gradient flow)
            y_hat_train_G[n, t, :] = y_hat

    # Normalize training loss by batch size and time steps
    loss /= (input_data_training.shape[0] * input_data_training.shape[1])

    # Backpropagate the loss
    loss.backward()

    # Update the model parameters
    optimizer.step()

    # Print training loss for this epoch
    print(f"Epoch: {epoch + 1} \t||\t Training Loss: {loss.item()}")
    LOSS[epoch] = loss.item()



#-----------------------------closedloop sysid of S through RENs------------------------
y_hat_train_S = torch.zeros(output_data_training.shape)
u_S = torch.zeros(input_data_training.shape)
# Training loop settings
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    optimizer.zero_grad()  # Reset the gradients before backpropagation
    loss = 0.0  # Initialize loss for this epoch

    # Training loop
    for n in range(input_data_training.shape[0]):  # Iterate over each batch
        REN.reset()  # Reset xi_ for each trajectory
        u_K = torch.zeros(input_dim)  # Reset the control input for each trajectory

        for t in range(input_data_training.shape[1]):  # Iterate over each time step
            u_ext = input_data_training[n, t, :]  # Extract external input
            u = (u_ext - u_K)  # Apply control input adjustment
            # Get model output
            y_hat = REN.forward(u.view(1, 1, -1))

            # Compute the control input
            u_K = controller.forward(y_hat.squeeze(0).squeeze(0))  # Update control law

            # Accumulate loss across all time steps and samples
            loss += MSE(output_data_training[n, t, :].view(1, 1, -1), y_hat)

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

    # Print training loss for this epoch
    print(f"Epoch: {epoch + 1} \t||\t Training Loss: {loss.item()}")
    LOSS[epoch] = loss.item()

# --------------------------PLOTS-----------------------------------

# Run open-loop simulation
# Generate input sequence (e.g., random noise)
u_seq = torch.randn(num_signals, horizon, input_dim) * 0.5  # Scaled random input

# Run open-loop simulation
x_traj = []
y_traj = []

for t in range(horizon):
    u_t = u_seq[:,t:t+1,:]
    x_traj.append(x.squeeze().item())  # Store state
    y_traj.append(x.squeeze().item())  # True output
    y_noisy_traj.append(y_noisy.squeeze().item())  # Noisy output
    x, y_noisy = sys.noisy_forward(x, u_t)  # Compute next state and noisy output

# Convert lists to tensors for plotting
x_traj = torch.tensor(x_traj)
y_traj = torch.tensor(y_traj)
y_noisy_traj = torch.tensor(y_noisy_traj)

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(y_traj, label="True Output", linestyle="dashed", color="b")
plt.plot(y_noisy_traj, label="Noisy Output", linestyle="solid", color="r", alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()
plt.title("Noisy Open-Loop Simulation")
plt.show()

# --------------Plot identification results for G-----------------
time_plot = np.arange(0, input_data_training.shape[1] * ts, ts)
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
axes[2].plot(time_plot, input_data_training[0, 0:len(time_plot), 0].detach().numpy(),
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
axes[2].plot(time_plot, u_S[0, 0:len(time_plot), 0].detach().numpy(),
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

REN.reset()  # Reset xi_ for each trajectory
y_traj = torch.zeros(horizon)

# Run open-loop simulation

for i in range(horizon-1):  # Iterate over each time step
    u_ext = input_data_training[0, i, :]  # Extract external input
    u = u_ext
    y_hat = REN.forward(u)
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