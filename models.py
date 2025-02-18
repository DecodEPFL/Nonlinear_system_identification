import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PointMassVehicle(nn.Module):
    def __init__(self, mass, sampling_time, drag_coefficient_1, drag_coefficient_2):
        super().__init__()
        self.mass = mass  # Mass of the vehicle
        self.Ts = sampling_time  # Sampling time
        self.b1 = drag_coefficient_1  # Drag coefficient 1
        self.b2 = drag_coefficient_2  # Drag coefficient 2
        #self.integral_error = torch.zeros(2)  # Initialize integral error for PI controller
        self.vehicle = True

    def drag_force(self, q):
        """Compute the drag force given the velocity q."""
        # Nonlinear drag: C(q) = b * |q|^2
        #drag = self.b2 * torch.norm(q) ** 2
        # Nonlinear drag: C(q) = b1 +  b2 * |q|
        drag = self.b1 + self.b2 * torch.norm(q)
        # Drag function: C(q) = b1 * |q| - b2 * tanh(|q|)
        #drag = (self.b1 * torch.norm(q) - self.b2 * torch.tanh(torch.norm(q)))
        return drag

    def forward(self, x, u):
        """
        Compute the next state and output of the system.

        Args:
            x (torch.Tensor): Current state (4D vector).
            u (torch.Tensor): Control input (2D vector).

        Returns:
            torch.Tensor, torch.Tensor: Next state and output of the system.
        """

        #Define position and velocity
        p = x[0:2]
        q = x[2:]

        # Compute drag force
        cq = self.drag_force(q)

        # Update equations based on the discrete-time model
        next_p = p + self.Ts * q
        next_q = q + self.Ts * (1 / self.mass) * (-cq*q + u)

        #Compute next state and output
        x = torch.cat((next_p, next_q), dim=0)
        y = next_p

        return x, y

class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vehicle = False

    def forward(self, x, u):
        """
        Compute the next state and output of the system.

        Args:
            x (torch.Tensor): Current state
            u (torch.Tensor): Control input

        Returns:
            torch.Tensor, torch.Tensor: Next state and output of the system.
        """

        #Compute next state and output
        x = x**2 + 1 + u
        y = x

        return x, y

class ProportionalController(nn.Module):
    def __init__(self, Kp, y_target):
        super().__init__()
        self.Kp = Kp
        self.y_target = y_target

    def forward(self, y):
        u = torch.matmul(self.Kp, self.y_target - y)
        return u

class NonLinearController(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y):
        u = -y**2 - 1 + 0.5*y
        return u

# Define a simple Multi-Layer Perceptron (MLP) class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define the model structure using nn.Sequential for easy layer stacking
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # First fully connected layer
            nn.SiLU(),  # Activation function after the first layer (Sigmoid Linear Unit)
            nn.Linear(hidden_size, hidden_size),  # Second fully connected layer (hidden layer)
            nn.ReLU(),  # Activation function after hidden layer (Rectified Linear Unit)
            nn.Linear(hidden_size, output_size)  # Output layer, no activation (raw scores)
        )

    def forward(self, x):
        # Check if the input tensor x is 3D (batch_size, sequence_length, input_size)
        if x.dim() == 3:
            # Extract dimensions
            batch_size, seq_length, input_size = x.size()

            # Flatten batch and sequence dimensions to process through MLP
            x = x.reshape(-1, input_size)  # Reshape to (batch_size * sequence_length, input_size)

            # Apply the MLP model to each feature vector
            x = self.model(x)  # Resulting shape: (batch_size * sequence_length, output_size)

            # Reshape back to original 3D shape (batch_size, sequence_length, output_size)
            output_size = x.size(-1)
            x = x.reshape(batch_size, seq_length, output_size)  # Reshape back to 3D
        else:
            # If x is not 3D, apply the MLP directly to x
            x = self.model(x)

        return x

# Define the PScan class for a parallel scan algorithm using PyTorch's autograd function
class PScan(torch.autograd.Function):
    # Given A (NxTx1) and X (NxTxD), this function expands A and X in parallel.
    # The algorithm is designed to perform in O(T) time complexity and
    # O(log(T)) time complexity if not core-bound.
    #
    # This helps compute the recurrence relation:
    # Y[:, 0] = Y_init
    # Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    #
    # This is equivalent to:
    # Y[:, t] = A[:, t] * Y_init + X[:, t]

    @staticmethod
    def expand_(A, X):
        # If A has only one time step, return as no expansion is needed
        if A.size(1) == 1:
            return
        # Handle the expansion for even T by splitting A and X into two parts
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)  # Reshape A for parallel processing
        Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)  # Reshape X similarly
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))  # Update X for next step
        Aa[:, :, 1].mul_(Aa[:, :, 0])  # Update A for next step
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])  # Recursive call for further expansion
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))  # Combine results for odd steps
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])  # Update A for the odd steps
        if T < A.size(1):
            X[:, -1].add_(A[:, -1].mul(X[:, -2]))  # Update the last element if T is odd
            A[:, -1].mul_(A[:, -2])  # Update the last A element

    @staticmethod
    def acc_rev_(A, X):
        # If X has only one time step, return as no accumulation is needed
        if X.size(1) == 1:
            return
        # Handle the reverse accumulation for even T
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1)  # Reshape for reverse accumulation
        Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1)  # Reshape X similarly
        Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))  # Accumulate X in reverse
        B = Aa[:, :, 0].clone()  # Clone A for the next step
        B[:, 1:].mul_(Aa[:, :-1, 1])  # Update B for reverse accumulation
        PScan.acc_rev_(B, Xa[:, :, 0])  # Recursive call for further accumulation
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))  # Combine results
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))  # Update the first element if T is odd

    # Forward pass: A is NxT, X is NxTxD, Y_init is NxD
    # Returns Y of the same shape as X with the recursive formula applied.
    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A[:, :, None].clone()  # Clone A for backpropagation
        ctx.Y_init = Y_init[:, None, :].clone()  # Clone Y_init for backpropagation
        ctx.A_star = ctx.A.clone()  # Clone A for further processing
        ctx.X_star = X.clone()  # Clone X for further processing
        PScan.expand_(ctx.A_star, ctx.X_star)  # Expand A and X for parallel computation
        return ctx.A_star * ctx.Y_init + ctx.X_star  # Compute the final output

    @staticmethod
    def backward(ctx, grad_output):
        # Compute gradients for backpropagation
        U = grad_output * ctx.A_star  # Multiply gradient with expanded A
        A = ctx.A.clone()  # Clone A for backpropagation steps
        R = grad_output.clone()  # Clone gradient for accumulation
        PScan.acc_rev_(A, R)  # Accumulate gradients in reverse order
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()  # Expand Y_init to match X_star
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])  # Compute intermediate values
        # Return the gradients for A, X, and Y_init
        return (Q * R).sum(-1), R, U.sum(dim=1)


# Create an instance of the PScan function for usage
pscan = PScan.apply

class LRU(nn.Module):
    # Implements a Linear Recurrent Unit (LRU) following the parametrization of
    # the paper "Resurrecting Linear Recurrences".
    # The LRU is simulated using Parallel Scan (fast!) when "scan" is set to True (default),
    # otherwise, it uses a recursive method (slow).
    def __init__(self, in_features, out_features, state_features, scan=True, rmin=0.9, rmax=1, max_phase=6.283):
        super().__init__()
        self.state_features = state_features  # Number of state features
        self.in_features = in_features  # Number of input features
        self.scan = scan  # Determines whether to use parallel scan or not
        self.out_features = out_features  # Number of output features

        # Define the linear transformation matrices D, B, and C
        self.D = nn.Parameter(torch.randn([out_features, in_features]) / math.sqrt(in_features))

        # Define random parameters for the complex eigenvalues (Lambda)
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        # Compute the modulus and the phase of the eigenvalues
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))

        # Initialize B and C as complex matrices for the state transitions and output calculations
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))

        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))

        # Initialize the state as a complex tensor
        self.state = torch.complex(torch.zeros(state_features), torch.zeros(state_features))

    def forward(self, input):
        # Ensure the state tensor is on the same device as B
        self.state = self.state.to(self.B.device)

        # Compute the eigenvalues in complex form
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im = Lambda_mod * torch.sin(torch.exp(self.theta_log))
        Lambda = torch.complex(Lambda_re, Lambda_im).to(self.state.device)

        # Compute gammas for the state update process
        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        gammas = gammas.to(self.state.device)

        # Prepare the output tensor with the appropriate dimensions
        output = torch.empty([i for i in input.shape[:-1]] + [self.out_features], device=self.B.device)

        # Input must be (Batches, Seq_length, Input size), otherwise adds dummy dimension = 1 for batches
        if input.dim() == 2:
            input = input.unsqueeze(0)

        if self.scan:
            # Simulate the LRU using Parallel Scan for faster computations
            input = input.permute(2, 1, 0)  # (Input size, Seq_length, Batches)

            # Prepare B matrix for broadcasting and multiplication
            B_unsqueezed = self.B.unsqueeze(-1).unsqueeze(-1)
            B_broadcasted = B_unsqueezed.expand(self.state_features, self.in_features, input.shape[1], input.shape[2])
            input_broadcasted = input.unsqueeze(0).expand(self.state_features, self.in_features, input.shape[1],
                                                          input.shape[2])

            # Elementwise multiplication and summation over the input dimension
            inputBU = torch.sum(B_broadcasted * input_broadcasted, dim=1)

            # Prepare matrix Lambda for scan
            Lambda = Lambda.unsqueeze(1)
            A = torch.tile(Lambda, (1, inputBU.shape[1]))

            # Apply Parallel Scan and get state sequence (initial condition = self.state)
            init = torch.complex(torch.zeros((self.state_features, inputBU.shape[2])),
                                 torch.zeros((self.state_features, inputBU.shape[2])))
            gammas_reshaped = gammas.unsqueeze(2)
            GBU = gammas_reshaped * inputBU

            # Apply the Parallel Scan to compute the states
            states = pscan(A, GBU, init)

            # Prepare the output calculation using matrices C and D
            C_unsqueezed = self.C.unsqueeze(-1).unsqueeze(-1)
            C_broadcasted = C_unsqueezed.expand(self.out_features, self.state_features, inputBU.shape[1],
                                                inputBU.shape[2])
            CX = torch.sum(C_broadcasted * states, dim=1)

            D_unsqueezed = self.D.unsqueeze(-1).unsqueeze(-1)
            D_broadcasted = D_unsqueezed.expand(self.out_features, self.in_features, input.shape[1], input.shape[2])
            DU = torch.sum(D_broadcasted * input, dim=1)

            # Compute the final output by combining the contributions from states and inputs
            output = 2 * CX.real + DU
            output = output.permute(2, 1, 0)  # Restore to (Batches, Seq length, Input size)
        else:
            # Simulate the LRU recursively, iterating over sequences
            for i, batch in enumerate(input):
                out_seq = torch.empty(input.shape[1], self.out_features)
                for j, step in enumerate(batch):
                    self.state = (Lambda * self.state + gammas * self.B @ step.to(dtype=self.B.dtype))
                    out_step = 2 * (self.C @ self.state).real + self.D @ step
                    out_seq[j] = out_step
                #self.state = torch.complex(torch.zeros_like(self.state.real), torch.zeros_like(self.state.real))
                output[i] = out_seq

        return output # Shape (Batches, Seq_length, Input size)


class SSM(nn.Module):
    # Implements LRU + a user-defined scaffolding, this is our SSM block.
    # Scaffolding can be modified. In this case, we have LRU, MLP plus a linear skip connection.
    def __init__(self, in_features, out_features, state_features, scan, mlp_hidden_size=30, rmin=0.9, rmax=1,
                 max_phase=6.283):
        super().__init__()
        self.mlp = MLP(out_features, mlp_hidden_size, out_features)  # MLP for additional non-linearity
        self.LRU = LRU(in_features, out_features, state_features, scan, rmin, rmax, max_phase)  # Linear Recurrent Unit
        self.model = nn.Sequential(self.LRU, self.mlp)  # Combine LRU and MLP in a sequential model
        self.lin = nn.Linear(in_features, out_features)  # Linear layer for skip connection

    def forward(self, input):
        # Compute the result by adding the output of the LRU-MLP model and the skip connection
        result = self.model(input) + self.lin(input)
        return result


class DeepLRU(nn.Module):
    # Implements a cascade of N SSMs. Linear pre- and post-processing can be modified.
    def __init__(self, N, in_features, out_features, mid_features, state_features, scan=True):
        super().__init__()
        self.linin = nn.Linear(in_features, mid_features)  # Linear input transformation
        self.linout = nn.Linear(mid_features, out_features)  # Linear output transformation

        # Create a list of SSM layers
        self.modelt = nn.ModuleList(
            [SSM(mid_features, mid_features, state_features, scan) for j in range(N)])

        # Insert the linear input transformation at the beginning and the output at the end
        self.modelt.insert(0, self.linin)
        self.modelt.append(self.linout)

        # Define the entire model as a sequential module
        self.model = nn.Sequential(*self.modelt)

    def forward(self, input):
        # Pass the input through the sequence of layers and return the result
        result = self.model(input)
        return result


# REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"
class PsiU(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.n = n
        self.n_xi = n_xi
        self.l = l
        self.m = m
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 0.1
        self.X = nn.Parameter((torch.randn(2*n_xi+l, 2*n_xi+l)*std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi)*std))
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n)*std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi)*std))
        self.D21 = nn.Parameter((torch.randn(m, l)*std))
        self.D22 = nn.Parameter((torch.randn(m, n)*std))
        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n)*std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi)
        self.B1 = torch.zeros(n_xi, l)
        self.E = torch.zeros(n_xi, n_xi)
        self.Lambda = torch.ones(l)
        self.C1 = torch.zeros(l, n_xi)
        self.D11 = torch.zeros(l, l)
        self.set_model_param()

    def set_model_param(self):
        n_xi = self.n_xi
        l = self.l
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2*n_xi+l)
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l)
        vec[0] = 1
        epsilon = torch.zeros(self.l)
        v = F.linear(xi, self.C1[0,:]) + F.linear(w, self.D12[0,:])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v/self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l)
            vec[i] = 1
            v = F.linear(xi, self.C1[i,:]) + F.linear(epsilon, self.D11[i,:]) + F.linear(w, self.D12[i,:])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v/self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon, self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + F.linear(w, self.D22)  # + self.bu
        return u, xi_


class PsiX(nn.Module):
    def __init__(self, f):
        super().__init__()
        n = 4
        m = 2
        self.f = f

    def forward(self, t, omega):
        y, u = omega
        psi_x = self.f(t, y, u)
        omega_ = 0
        return psi_x, omega_


class Controller(nn.Module):
    def __init__(self, f, n, m, n_xi, l):
        super().__init__()
        self.n = n
        self.m = m
        self.psi_x = PsiX(f)
        self.psi_u = PsiU(self.n, self.m, n_xi, l)

    def forward(self, t, y_, xi, omega):
        psi_x, _ = self.psi_x(t, omega)
        w_ = y_ - psi_x
        u_, xi_ = self.psi_u(t, w_, xi)
        omega_ = (y_, u_)
        return u_, xi_, omega_