from timeit import repeat
import torch
import torch.nn as nn

class NonLinearModel(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x, u):
        """
        Computse the next state and output of the system.

        Args:
            x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim) ---> output feedback, state not accessible
            u (torch.Tensor): plant's input at t. shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Next state and output of the system at t+1. shape = (batch_size, 1, state_dim), shape = (batch_size, 1, output_dim)
        """

        #Compute next state and output
        x = x**2 + 1 + u
        y = x

        return x, y

    def noisy_forward(self, x, u, output_noise_std):
        """
        Computes the next state and output with noise.

        Args:
            x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim) ---> output feedback, state not accessible
            u (torch.Tensor): plant's input at t. shape = (batch_size, 1, input_dim)
            output_noise_std: standard deviation of noise
        Returns:
            torch.Tensor, torch.Tensor: Noisy next state and output at t+1.
        """
        x, y = self.forward(x, u)

        # Add Gaussian additive noise
        noise = torch.randn_like(y) * output_noise_std
        y_noisy = y + noise

        return x, y_noisy

    #TODO: improve code for initial conditions
    def run(self, x0, u_ext, output_noise_std):
        """
        Simulates the open-loop system for a given initial condition and external signal.

        Args:
            x0 (torch.Tensor): Initial state.
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim) [u0,...,u_T-1]
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs  [y0, ..., y_T-1]
        """
        horizon = u_ext.shape[1]
        batch_size = u_ext.shape[0]
        if x0.shape == (batch_size, 1, self.state_dim):
            x0_batched = x0
        elif x0.shape == (1, 1, self.state_dim):
            x0_batched = x0.expand(batch_size, 1, self.state_dim)
        elif x0.shape == torch.Size([1]):
            x0_batched = x0.view(1,1,-1).expand(batch_size, 1, self.state_dim)
        else:
            print('Wrong shape of initial conditions')
            x0_batched = None

        # Storage for trajectories
        y_traj = torch.zeros((batch_size, horizon, self.output_dim))

        # Compute initial output with noisy measurements
        y0_batched = x0_batched
        y0_noisy_batched = y0_batched + torch.randn_like(y0_batched) * output_noise_std

        # Initialize state
        x = x0_batched.clone()
        y = y0_noisy_batched.clone()

        for t in range(horizon):
            y_traj[:, t:t + 1, :] = y  # Store output
            x, y = self.noisy_forward(x, u_ext[:, t:t+1, :], output_noise_std)  # Apply input to plant
        return y_traj

    def __call__(self, x0, u_ext, output_noise_std):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        return self.run(x0, u_ext, output_noise_std)

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
            torch.Tensor, torch.Tensor: Next control input at t. shape = (batch_size, 1, output_K_dim)
        """
        u = -y**2 - 1 + 0.5*y
        return u


class ClosedLoopSystem(nn.Module):
    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, system_model, controller, negative: bool = False):
        super().__init__()
        self.system_model = system_model
        self.controller = controller
        self.negative = negative

        if hasattr(self.system_model, "noisy_forward"):
            self.system_model_tipe = "real_sys"
        # elif hasattr(self.system_model, "controller"):
        #     self.system_model_tipe = "closed_loop_REN"
        else:
            self.system_model_tipe = "REN"

    def forward(self, y, u_ext):
        """
        Compute the next state and output of the system.

        Args:
            u_ext (torch.Tensor): external input at t. shape = (batch_size, 1, input_dim)
            y (torch.Tensor): plant's output at t. shape = (batch_size, 1, output_dim)

        Returns:
            torch.Tensor, torch.Tensor: Input of plant and next output at t+1. shape = (batch_size, 1, state_dim), shape = (batch_size, 1, output_dim)
        """

        #Compute next state and output
        control_u = self.controller.forward(y)  # Compute control input
        u = control_u + u_ext
        x = y
        if self.system_model_tipe == "real_sys":
            x, y = self.system_model.forward(x, u)
        elif self.system_model_tipe == "REN":
            y = self.system_model.forward(u)
        return u, y

    def noisy_forward(self, y, u_ext, output_noise_std):
        """
        Compute the next state and output of the system.

        Args:
            u_ext (torch.Tensor): external input at t. shape = (batch_size, 1, input_dim)
            y (torch.Tensor): plant's output at t. shape = (batch_size, 1, output_dim)
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor, torch.Tensor: Input of plant and next output at t+1. shape = (batch_size, 1, state_dim), shape = (batch_size, 1, output_dim)
        """

        u, y = self.forward(y, u_ext)

        # Add Gaussian additive noise
        noise = torch.randn_like(y) * output_noise_std
        y_noisy = y + noise

        return u, y_noisy


    def run(self, x0, u_ext, output_noise_std):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim)
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """

        batch_size = u_ext.shape[0]
        horizon = u_ext.shape[1]

        if self.system_model_tipe == "real_sys":
            output_dim = self.system_model.output_dim
            input_dim = self.system_model.input_dim
            state_dim = self.system_model.state_dim

            if x0.shape == (batch_size, 1, state_dim):
                x0_batched = x0
            elif x0.shape == (1, 1, state_dim):
                x0_batched = x0.expand(batch_size, 1, state_dim)
            elif x0.shape == torch.Size([1]):
                x0_batched = x0.view(1,1,-1).expand(batch_size, 1, state_dim)
            else:
                print('Wrong shape of initial conditions')
                x0_batched = None

            # Compute initial output with noisy measurements
            y0_batched = x0_batched
            y0_noisy_batched = y0_batched + torch.randn_like(y0_batched) * output_noise_std

            # Initialize state
            x = x0_batched.clone()
            y = y0_noisy_batched.clone()

            # Storage for trajectories
            y_traj = []
            u_traj = []

            for t in range(horizon):
                y_traj.append(y)  # Store output
                u, y = self.noisy_forward(y, u_ext[:, t:t + 1, :], output_noise_std)  # Apply input to plant
                u_traj.append(u)  # Store input

            # Convert lists to tensors
            y_traj = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)
            u_traj = torch.cat(u_traj, dim=1)  # Shape: (batch_size, horizon, input_dim)

        elif self.system_model_tipe == "REN":
            output_dim = self.system_model.dim_out
            input_dim = self.system_model.dim_in

            self.system_model.reset()
            y = self.system_model.y_init.detach().clone().repeat(batch_size, 1, 1)

            # Storage for trajectories
            y_traj = []
            u_traj = []

            for t in range(horizon):
                control_u = self.controller.forward(y)  # Compute control input

                #minus sign for the control input
                if self.negative:
                    control_u = -control_u
                u = control_u + u_ext[:, t:t + 1, :]

                y = y + torch.randn_like(y) * output_noise_std
                y_traj.append(y)  # Store output
                u_traj.append(u)  # Store input
                y = self.system_model.forward(u)  # Apply input to REN


            # Convert lists to tensors
            y_traj = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)
            u_traj = torch.cat(u_traj, dim=1)  # Shape: (batch_size, horizon, input_dim)
        else:
            output_dim = self.system_model.sys.dim_out
            input_dim = self.system_model.sys.dim_in

            self.system_model.sys.reset()
            y = self.system_model.sys.y_init.detach().clone().repeat(batch_size, 1, 1)
            # Storage for trajectories
            y_traj = []
            u_traj = []

            for t in range(horizon):
                control_u = self.controller.forward(y)  # Compute control input

                u = control_u + u_ext[:, t:t + 1, :]
                y_traj.append(y)  # Store output
                u_traj.append(u)  # Store input
                y = system_model(u)

            # Convert lists to tensors
            y_traj = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)
            u_traj = torch.cat(u_traj, dim=1)  # Shape: (batch_size, horizon, input_dim)

        return u_traj, y_traj

    def __call__(self, x0, u_ext, output_noise_std):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        return self.run(x0, u_ext, output_noise_std)