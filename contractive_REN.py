import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


class ContractiveREN(nn.Module):
    """
    Acyclic contractive recurrent equilibrium network, following the paper:
    "Recurrent equilibrium networks: Flexible dynamic models with guaranteed
    stability and robustness, Revay M et al. ."

    The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
    The model is described as,

                    [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                    [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_w ]
                    [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_u ]

    where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
    are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

    NOTE: REN has input "u", output "y", and internal state "x". When used in closed-loop,
          the REN input "u" would be the noise reconstruction ("w") and the REN output ("y")
          would be the input to the plant. The internal state of the REN ("x") should not be mistaken
          with the internal state of the plant.
    """

    def __init__(
        self, input_dim: int, output_dim: int, state_dim: int,
        dim_nl: int, initialization_std: float = 0.5, pos_def_tol: float = 0.001, contraction_rate_lb: float = 1.0
    ):
        """
        Args:
            input_dim (int): Input (u) dimension.
            output_dim (int): Output (y) dimension.
            state_dim (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
        """
        super().__init__()

        # set dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.dim_nl = dim_nl

        # set functionalities
        self.contraction_rate_lb = contraction_rate_lb

        # auxiliary elements
        self.epsilon = pos_def_tol

        # define matrices shapes
        # auxiliary matrices
        self.X_shape = (2 * self.state_dim + self.dim_nl, 2 * self.state_dim + self.dim_nl)
        self.Y_shape = (self.state_dim, self.state_dim)
        # nn state dynamics
        self.B2_shape = (self.state_dim, self.input_dim)
        # nn output
        self.C2_shape = (self.output_dim, self.state_dim)
        self.D21_shape = (self.output_dim, self.dim_nl)
        self.D22_shape = (self.output_dim, self.input_dim)
        # v signal
        self.D12_shape = (self.dim_nl, self.input_dim)

        # define trainable params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        self._init_trainable_params(initialization_std)

        # mask
        self.register_buffer('eye_mask_H', torch.eye(2 * self.state_dim + self.dim_nl))
        self.register_buffer('eye_mask_w', torch.eye(self.dim_nl))

        # Initialize internal state to None by default
        self.register_buffer('x', None)

        # # initialize internal state
        # if internal_state_init is None:
        #     if y_init is None:
        #         self.x = torch.zeros(1, 1, self.state_dim)
        #     else:
        #         y_init = y_init.reshape(1, -1)
        #         self.x = torch.linalg.lstsq(self.C2, y_init.to(self.C2.device).squeeze(1).T)[0].T
        # else:
        #     assert isinstance(internal_state_init, torch.Tensor)
        #     self.x = internal_state_init.reshape(1, 1, self.state_dim)
        # self.register_buffer('x_init', self.x.detach().clone())
        # self.register_buffer('y_init', F.linear(self.x_init, self.C2))

    def _update_model_param(self):
        """
        Update non-trainable matrices according to the REN formulation to preserve contraction.
        """
        # dependent params
        H = torch.matmul(self.X.T, self.X) + self.epsilon * self.eye_mask_H
        h1, h2, h3 = torch.split(H, [self.state_dim, self.dim_nl, self.state_dim], dim=0)
        H11, H12, H13 = torch.split(h1, [self.state_dim, self.dim_nl, self.state_dim], dim=1)
        H21, H22, _ = torch.split(h2, [self.state_dim, self.dim_nl, self.state_dim], dim=1)
        H31, H32, H33 = torch.split(h3, [self.state_dim, self.dim_nl, self.state_dim], dim=1)
        P = H33

        # nn state dynamics
        self.F = H31
        self.B1 = H32

        # nn output
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)
        self.E_inv = self.E.inverse()

        # v signal for strictly acyclic REN
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, u_in):
        """
        Forward pass of REN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.input_dim).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.output_dim).
        """
        # update non-trainable model params
        self._update_model_param()

        batch_size = u_in.shape[0]

        w = torch.zeros(batch_size, 1, self.dim_nl, device=u_in.device)

        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.dim_nl):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i,:])
            w = w + (self.eye_mask_w[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.dim_nl)

        # compute next state using Eq. 18
        self.x = F.linear(F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2), self.E_inv)

        # compute output
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)
        return y_out

    #compute initial internal state to have the output match the output of the system
    def compute_x0_to_match_sys_output(self, x0_sys):
        y0_sys = x0_sys
        #y_init = y_init.reshape(1, -1)
        x0 = torch.linalg.lstsq(self.C2, y0_sys.to(self.C2.device).squeeze(1).T)[0].T.unsqueeze(1)

        return x0

    def reset(self, x0_sys=None, batch_size=1):
        """
        Reset the internal state.

        Args:
            x0_sys (torch.Tensor, optional): Initial state of the real plant, shape = (batch_size, 1, state_dim).
            batch_size (int): Batch size for initialization.
        """

        sys_state_dim = x0_sys.shape[-1]

        if x0_sys is None:
            x0_sys = torch.zeros(1, 1, sys_state_dim)  # Default to zeros
        else:
            if x0_sys.shape == (batch_size, 1, sys_state_dim):
                x0_sys = x0_sys
            elif x0_sys.shape == (1, 1, sys_state_dim):
                x0_sys = x0_sys.expand(batch_size, 1, sys_state_dim)
            elif x0_sys.shape == torch.Size([sys_state_dim]):
                x0_sys = x0_sys.view(1, 1, -1).expand(batch_size, 1, sys_state_dim)
            #x0_sys = x0_sys.view(-1, 1, sys_state_dim)  # Ensure shape (batch_size, 1, state_dim)
            #x0_sys = x0_sys.expand(batch_size, -1, -1)  # Expand if needed
            # x0_sys = x0_sys.view(-1, 1, sys_state_dim)  # Ensure shape (batch_size, 1, sys_state_dim)
            # if x0_sys.shape[0] == 1:  # If it's a single sample, expand it for batch processing
            #     x0_sys = x0_sys.expand(batch_size, 1, sys_state_dim)
        x0 = self.compute_x0_to_match_sys_output(x0_sys)
        self.x = x0

    def y0_from_x0(self, x0):
        y0 = F.linear(x0, self.C2)
        return y0


    def run(self, x0_sys, u_in):
        """
        Runs the forward pass of REN for a whole input sequence of length horizon.

        Args:
            x0_sys: initial condition of the real plant
            u_in (torch.Tensor): Input with the size of (batch_size, horizon, self.input_dim).

        Return:
            y_out (torch.Tensor): Output with (batch_size, horizon, self.output_dim).
        """
        horizon = u_in.shape[1]
        batch_size = u_in.shape[0]

        # Storage for trajectories
        y_traj = []

        self.reset(x0_sys = x0_sys, batch_size = batch_size)

        y0 = self.y0_from_x0(self.x)
        y = y0

        for t in range(horizon):
            y_traj.append(y)  # Store output
            y = self.forward(u_in[:, t:t + 1, :])
            # note that the last input is not used

        y_traj = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)

        return y_traj

    # init trainable params
    def _init_trainable_params(self, initialization_std):
        for training_param_name in self.training_param_names:  # name of one of the training params, e.g., X
            # read the defined shapes of the selected training param, e.g., X_shape
            shape = getattr(self, training_param_name + '_shape')
            # define the selected param (e.g., self.X) as nn.Parameter
            setattr(self, training_param_name, nn.Parameter((torch.randn(*shape) * initialization_std)))

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict

    def __call__(self, x0_sys, u_in):
        return self.run(x0_sys, u_in)