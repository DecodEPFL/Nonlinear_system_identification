from timeit import repeat
import torch
import torch.nn as nn
from debugpy.server.cli import in_range
import torch.nn.functional as F


class RobotsSystem(torch.nn.Module):
    def __init__(self, m: float = 1.0, ts: float = 0.05,
                 b: float = 1, b2: float = 0.1,
                 n_agents: int = 1, state_dim_agent: int = 4, input_dim_agent: int = 2,
                 output_dim_agent: int = 4, linearize: bool = False):
        """
        Initializes the robot system model.

        Args:
            m (float): Mass of each agent (robot).
            ts (float): Discrete time step used for system updates.
            b (float): Linear drag coefficient (used in both linear and nonlinear models).
            b2 (float): Nonlinear drag coefficient (used only in nonlinear models).
            n_agents (int): Number of agents (robots) in the system.
            state_dim_agent (int): Number of state variables per agent. Default is 4 (position and velocity on both x and y axes).
            input_dim_agent (int): Number of control input variables per agent. Default is 2 (forces applied to the agent along x and y axes).
            output_dim_agent (int): Number of observable output variables per agent.
                                    Defines what part of the state is measurable. Default is 2 (positions along x and y axes only).
            linearize (bool): If True, uses a linearized model of the system.
                              Otherwise, a nonlinear model is used where friction depends on speed.

        """
        super().__init__()
        self.m = m  # Mass of the vehicle
        self.ts = ts  # Sampling time
        self.b = b  # Drag coefficient 1
        self.b2 = b2  # Drag coefficient 2
        self.n_agents = n_agents
        self.state_dim_agent = state_dim_agent
        self.input_dim_agent = input_dim_agent
        self.output_dim_agent = output_dim_agent
        self.linearize = linearize

        self.state_dim = self.state_dim_agent * self.n_agents
        self.input_dim = self.input_dim_agent * self.n_agents
        self.output_dim = self.output_dim_agent * self.n_agents


        self.tanh_nonlinearity = False

        #shape of B is (state_dim, input_dim)
        Bi = torch.tensor([[0, 0],[0, 0],[1/m, 0], [0, 1/m]]) * self.ts
        B = torch.kron(torch.eye(self.n_agents), Bi)
        assert B.shape == (self.state_dim, self.input_dim)
        self.register_buffer('B', B)

        Ai = torch.eye(self.state_dim_agent)
        Ai[0, 2] = self.ts
        Ai[1, 3] = self.ts

        A = torch.kron(torch.eye(self.n_agents), Ai)

        assert A.shape == (self.state_dim, self.state_dim)
        self.register_buffer('A', A)

        # Initialize internal state to None by default
        self.register_buffer('x', None)

    def reset(self, x0=None, batch_size=1):
        """
        Reset the internal state.

        Args:
            x0 (torch.Tensor, optional): Initial state, shape = (batch_size, 1, state_dim).
            batch_size (int): Batch size for initialization.
        """
        if x0 is None:
            x0 = torch.zeros(batch_size, 1, self.state_dim)  # Default to zeros
        else:
            # Ensure x0 has the correct shape
            x0 = x0.view(-1, 1, self.state_dim)  # Ensure shape (batch_size, 1, state_dim)
            x0 = x0.expand(batch_size, -1, -1)  # Expand if needed

        self.x = x0

    @staticmethod
    def y0_from_x0(x0):
        y0 = x0
        return y0

    def drag_force(self, x):
        """
        Compute the drag force given the state

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)

        Returns:
            - drag (torch.Tensor): transformation of the state C(x) representing the drag force shape = (batch_size, 1, state_dim)
        """
        # Nonlinear drag: C(q) = b * |q|^2 (squared) ???

        batch_size, _, state_dim = x.shape
        # Reshape to separate agents: (batch_size, 1, n_agents, state_dim_agent)
        x_reshaped = x.view(batch_size, 1, self.n_agents, self.state_dim_agent)

        # Extract the first two components (they remain unchanged)
        p = torch.zeros_like(x_reshaped[..., :2])  # shape: (batch_size, 1, n_agents, 2)

        # Extract the last two components (q1, q2) and compute their norm
        q = x_reshaped[..., 2:]  # shape: (batch_size, 1, n_agents, 2)
        q_norm = torch.norm(q, dim=-1, keepdim=True)  # shape: (batch_size, 1, n_agents, 1)

        if self.linearize:
            q_new = -self.ts / self.m * self.b * q
        elif self.tanh_nonlinearity:
            q_new = -self.ts / self.m * self.b * q -self.ts / self.m *self.b2 * torch.tanh(q)
        else:
            # Update q components
            q_new = -self.ts / self.m * self.b * q - self.ts / self.m *self.b2 * q_norm * q

        # Concatenate the unchanged p and updated q
        drag = torch.cat((p, q_new), dim=-1)  # shape: (batch_size, 1, n_agents, 4)

        # Reshape back to (batch_size, 1, state_dim)
        drag = drag.view(batch_size, 1, state_dim)

        return drag

    def forward(self, u):
        """
        Compute the next output of the system.

        Args:
            u (torch.Tensor): Plant's input at t. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor: Output of the system at t+1. Shape = (batch_size, 1, output_dim)
        """
        if self.x is None:
            raise ValueError("State not initialized. Call `reset()` before using forward().")

        # Compute next state and update internal state
        x1 = F.linear(self.x, self.A)
        x2 = F.linear(u, self.B)
        x3 = self.drag_force(self.x)
        self.x = F.linear(self.x, self.A) + F.linear(u, self.B) + self.drag_force(self.x)
        y = self.x  # Output is the updated state

        return y

    def run(self, x0, u_ext, output_noise=None):
        """
        Simulates the open-loop system for a given initial condition and external signal.

        Args:
            x0 (torch.Tensor): Initial state.
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim) [u0,...,u_T-1]
            output_noise: realization of output noise

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim) [y0, ..., y_T-1]
        """
        horizon = u_ext.shape[1]
        batch_size = u_ext.shape[0]

        # Storage for trajectories
        y_traj = []

        self.reset(x0=x0, batch_size=batch_size)

        if output_noise is None:
            output_noise = torch.zeros(batch_size, horizon, self.output_dim)

        # Compute initial output with noisy measurements
        y0 = self.y0_from_x0(self.x)
        y = y0 + output_noise[:, 0:1, :]  # First noise realization

        for t in range(horizon):
            y_traj.append(y) # Store output
            #noisy forward
            y = self.forward(u_ext[:, t:t+1, :])  + output_noise[:, t + 1:t + 2, :]

        y_traj = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)

        return y_traj

    def __call__(self, x0, u_ext, output_noise):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim)

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim) [y0, ..., y_T-1]
        """
        return self.run(x0, u_ext, output_noise)

class Proportional_Controller(nn.Module):
    def __init__(self, input_k_dim = 4, output_k_dim = 2, n_agents = 1, kp=None):
        super().__init__()
        self.input_k_dim = input_k_dim #=output_dim of sys
        self.output_k_dim = output_k_dim #=input_dim of sys
        self.n_agents = n_agents

        if kp is None:
            self.kp = torch.kron(torch.eye(self.n_agents), torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32))
        else:
            self.kp = kp

    def forward(self, y):
        """
        Args:
            y (torch.Tensor): plant's output (controller's input) at t. shape = (batch_size, 1, input_K_dim)

        Returns:
            torch.Tensor, torch.Tensor: Next control input at t. shape = (batch_size, 1, output_K_dim)
        """
        #y_target = 0
        u = F.linear(-y, self.kp)
        return u

class NonLinearModel(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize internal state to None by default
        self.register_buffer('x', None)

    def reset(self, x0=None, batch_size=1):
        """
        Reset the internal state.

        Args:
            x0 (torch.Tensor, optional): Initial state, shape = (batch_size, 1, state_dim).
            batch_size (int): Batch size for initialization.
        """
        if x0 is None:
            x0 = torch.zeros(batch_size, 1, self.state_dim)  # Default to zeros
        else:
            # Ensure x0 has the correct shape
            x0 = x0.view(-1, 1, self.state_dim)  # Ensure shape (batch_size, 1, state_dim)
            x0 = x0.expand(batch_size, -1, -1)  # Expand if needed

        self.x = x0

    @staticmethod
    def y0_from_x0(x0):
        y0 = x0
        return y0

    def forward(self, u):
        """
        Compute the next output of the system.

        Args:
            u (torch.Tensor): Plant's input at t. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor: Output of the system at t+1. Shape = (batch_size, 1, output_dim)
        """
        if self.x is None:
            raise ValueError("State not initialized. Call `reset()` before using forward().")

        # Compute next state and update internal state
        self.x = self.x**2 + 1 + u
        y = self.x  # Output is the updated state

        return y

    def run(self, x0, u_ext, output_noise=None):
        """
        Simulates the open-loop system for a given initial condition and external signal.

        Args:
            x0 (torch.Tensor): Initial state.
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim) [u0,...,u_T-1]
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim) [y0, ..., y_T-1]
        """
        horizon = u_ext.shape[1]
        batch_size = u_ext.shape[0]

        # Storage for trajectories
        y_traj = []

        self.reset(x0=x0, batch_size=batch_size)

        if output_noise is None:
            output_noise = torch.zeros(batch_size, horizon, self.output_dim)

        # Compute initial output with noisy measurements
        y0 = self.y0_from_x0(self.x)
        y = y0 + output_noise[:, 0:1, :]

        for t in range(horizon):
            y_traj.append(y) # Store output
            #noisy forward
            y = self.forward(u_ext[:, t:t+1, :])  + output_noise[:, t + 1:t + 2, :]

        y_traj = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)

        return y_traj

    def __call__(self, x0, u_ext, output_noise):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim)

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim) [y0, ..., y_T-1]
        """
        return self.run(x0, u_ext, output_noise)

class NonLinearController(nn.Module):
    def __init__(self, input_k_dim, output_k_dim):
        super().__init__()
        self.input_k_dim = input_k_dim #=output_dim of sys
        self.output_k_dim = output_k_dim #=input_dim of sys

    @staticmethod
    def forward(y):
        """
        Args:
            y (torch.Tensor): plant's output (controller's input) at t. shape = (batch_size, 1, input_K_dim)

        Returns:
            torch.Tensor, torch.Tensor: Next control input at t. shape = (batch_size, 1, output_K_dim)
        """
        u = -y**2 - 1 + 0.5*y
        return u

#TODO
class ClosedLoopSystem(nn.Module):
    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, system_model, controller, negative: bool = False):
        super().__init__()
        self.system_model = system_model
        self.controller = controller
        self.negative = negative
        self.output_dim = self.system_model.output_dim
        self.input_dim = self.system_model.input_dim
        self.state_dim = self.system_model.state_dim

        self.register_buffer('x', None)
        self.register_buffer('y_prev', None)

    def reset(self, x0=None, batch_size=1):
        """
        Reset the internal state.

        Args:
            x0 (torch.Tensor, optional): Initial state, shape = (batch_size, 1, state_dim).
            batch_size (int): Batch size for initialization.
        """

        self.system_model.reset(x0, batch_size)
        self.x = self.system_model.x
        y0 = self.y0_from_x0(self.system_model.x)
        self.y_prev = y0

    def y0_from_x0(self, x0):
        y0 = self.system_model.y0_from_x0(x0)
        return y0

    def forward(self, u_ext):
        """
        Compute the next output of the system.

        Args:
            u_ext (torch.Tensor): external input at t. shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor: Next output at t+1. shape = (batch_size, 1, output_dim)
        """

        #Compute next state and output
        control_u = self.controller.forward(self.y_prev)  # Compute control input
        # minus sign for the control input
        if self.negative:
            control_u = -control_u
        u = control_u + u_ext
        y = self.system_model.forward(u)
        self.y_prev = y
        return y


    def run(self, x0, u_ext, output_noise=None):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim)
            output_noise: realizations of output noise

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim) [y0, ..., y_T-1]
        """

        batch_size = u_ext.shape[0]
        horizon = u_ext.shape[1]

        # Storage for trajectories
        y_traj = []
        u_traj = []
        if output_noise is None:
            output_noise = torch.zeros(batch_size, horizon, self.output_dim)
        self.system_model.reset(x0, batch_size)
        # Use pre-generated noise
        y0 = self.y0_from_x0(self.system_model.x)
        y = y0 + output_noise[:, 0:1, :]  # First noise realization
        self.y_prev = y

        for t in range(horizon):
            y_traj.append(y)  # Store output
            control_u = self.controller.forward(self.y_prev)  # Compute control input
            if self.negative:
                control_u = -control_u
            u = control_u + u_ext[:, t:t + 1, :]  # Compute total input to plant
            u_traj.append(u)  # Store input trajectory

            # Use pre-generated noise
            y = self.system_model.forward(u) + output_noise[:, t + 1:t + 2, :]
            self.y_prev = y

        # Convert lists to tensors
        y_traj = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)
        u_traj = torch.cat(u_traj, dim=1)  # Shape: (batch_size, horizon, input_dim)

        return u_traj, y_traj

    def __call__(self, x0, u_ext, output_noise):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim)
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim) [y0, ..., y_T-1]
        """
        return self.run(x0, u_ext, output_noise)


class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1, num_layers=1):
        super(RNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_dim))  # Learnable h0

        # RNN layer
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def reset(self, x0_sys=None, batch_size=1):
        """
        Reset the internal state.

        Args:
            x0_sys (torch.Tensor, optional): Initial state of the real plant, shape = (batch_size, 1, state_dim).
            batch_size (int): Batch size for initialization.
        """


    def y0_from_x0(self, x0):
        y0 = F.linear(x0, self.C2)
        return y0

    def forward(self, u_in):
        """
        Forward pass of RNN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.input_dim).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.output_dim).
        """
        h0 = self.h0.expand(-1, 2, -1)
        out, _ = self.rnn(u_in, h0)  # Forward through RNN
        y_out = self.fc(out)  # Linear output layer
        return y_out

    def run(self, x0_sys, u_in):
        """
        Runs the forward pass of RNN for a whole input sequence of length horizon.

        Args:
            x0_sys: Initial condition of the real plant (not used explicitly in RNN).
            u_in (torch.Tensor): Input with the size of (batch_size, horizon, self.input_dim).

        Return:
            y_out (torch.Tensor): Output with (batch_size, horizon, self.output_dim).
        """
        h0 = self.h0.expand(-1, 2, -1)  # Initialize hidden state
        out, _ = self.rnn(u_in, h0)  # Process full sequence
        y_out = self.fc(out)  # Apply final layer
        return y_out

    def __call__(self, x0_sys, u_in):
        return self.run(x0_sys, u_in)


class RNNSystemIdentificationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1):
        super(RNNSystemIdentificationModel, self).__init__()

        # Define the RNN layer
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        # Output layer that maps hidden states to output dimension
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, horizon, input_dim)

        Returns:
            output (Tensor): Output tensor of shape (batch_size, horizon, output_dim)
        """
        # Pass input through RNN
        rnn_out, _ = self.rnn(x)  # (batch_size, horizon, hidden_dim)

        # Pass the RNN outputs through a linear layer to get the desired output_dim
        output = self.output_layer(rnn_out)  # (batch_size, horizon, output_dim)

        return output