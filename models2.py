from timeit import repeat

import torch
import torch.nn as nn

class NonLinearModel(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, output_noise_std):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_noise_std = output_noise_std

        # Noise standard deviation
        self.output_noise_std = output_noise_std

    def forward(self, x, u):
        """
        Compute the next state and output of the system.

        Args:
            x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            u (torch.Tensor): Control input at t. shape = (batch_size, 1, input_dim)
            y (torch.Tensor): plant's output at t. shape = (batch_size, 1, output_dim)

        Returns:
            torch.Tensor, torch.Tensor: Next state and output of the system. shape = (batch_size, 1, state_dim), shape = (batch_size, 1, output_dim)
        """

        #Compute next state and output
        x = x**2 + 1 + u
        y = x

        return x, y

    def noisy_forward(self, x, u):
        """
        Compute the next state and output with noise.

        Args:
            x (torch.Tensor): State at t. Shape = (batch_size, 1, state_dim)
            u (torch.Tensor): Control input at t. Shape = (batch_size, 1, input_dim)
            y (torch.Tensor): plant's output at t. shape = (batch_size, 1, output_dim)
        Returns:
            torch.Tensor, torch.Tensor: Noisy next state and output.
        """
        x, y = self.forward(x, u)

        # Add Gaussian additive noise
        noise = torch.randn_like(y) * self.output_noise_std
        y_noisy = y + noise

        return x, y_noisy

    def run(self, x0, u_ext, horizon):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)
            horizon (int): Number of time steps to simulate

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """

        batch_size = u_ext.shape[0]
        if x0.shape == (batch_size, 1, self.state_dim):
            x0_batched = x0
        elif x0.shape == (1, 1, self.state_dim):
            x0_batched = x0.expand(batch_size, 1, self.state_dim)
        elif x0.shape == torch.Size([1]):
            x0_batched = x0.view(1,1,-1).expand(batch_size, 1, self.state_dim)
        else:
            print('Wrong shape of initial conditions')

        # Storage for trajectories
        y_traj = torch.zeros((batch_size, horizon, self.output_dim))

        # Compute initial output with noisy measurements
        y0_batched = x0_batched
        y0_noisy_batched = y0_batched + torch.randn_like(y0_batched) * self.output_noise_std

        # Initialize state
        x = x0_batched.clone()
        y = y0_noisy_batched.clone()

        for t in range(horizon):
            y_traj[:, t:t+1, :] = y  # Store output
            x, y = self.noisy_forward(x, u_ext[:, t:t+1, :])  # Apply input to plant
        return y_traj

    def __call__(self, x0, u_ext, horizon):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        return self.run(x0, u_ext, horizon)

class NonLinearController(nn.Module):
    def __init__(self, input_K_dim, output_K_dim):
        super().__init__()
        self.input_K_dim = input_K_dim #=output_dim of sys
        self.output_K_dim = output_K_dim #=input_dim of sys

    def forward(self, y):
        """
        Args:
            y (torch.Tensor): plant's output (controller's input) at t. shape = (batch_size, 1, input_K_dim)

        Returns:
            torch.Tensor, torch.Tensor: Next control input. shape = (batch_size, 1, output_K_dim)
        """
        u = -y**2 - 1 + 0.5*y
        return u


class ClosedLoopSystem(nn.Module):
    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, sys, controller):
        super().__init__()
        self.sys = sys
        self.controller = controller

    def run(self, x0, u_ext, horizon):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """

        output_dim = self.sys.output_dim
        input_dim = self.sys.input_dim
        state_dim = self.sys.state_dim

        batch_size = u_ext.shape[0]
        if x0.shape == (batch_size, 1, state_dim):
            x0_batched = x0
        elif x0.shape == (1, 1, state_dim):
            x0_batched = x0.expand(batch_size, 1, state_dim)
        elif x0.shape == torch.Size([1]):
            x0_batched = x0.view(1,1,-1).expand(batch_size, 1, state_dim)
        else:
            print('Wrong shape of initial conditions')


        # Storage for trajectories
        y_traj = torch.zeros((batch_size, horizon, output_dim))
        u_traj = torch.zeros((batch_size, horizon, input_dim))

        # Compute initial output with noisy measurements
        y0_batched = x0_batched
        y0_noisy_batched = y0_batched + torch.randn_like(y0_batched) * self.sys.output_noise_std
        # Initialize state
        x = x0_batched.clone()
        y = y0_noisy_batched.clone()

        for t in range(horizon):
            control_u = self.controller.forward(y)  # Compute control input
            u = control_u + u_ext[:, t:t+1, :]

            y_traj[:, t:t+1, :] = y  # Store output
            u_traj[:, t:t+1, :] = u  # Store input

            x, y = self.sys.noisy_forward(x, u)  # Apply input to plant

        return u_traj, y_traj

    def __call__(self, x0, u_ext, horizon):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        return self.run(x0, u_ext, horizon)


