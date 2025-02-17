import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from scan_utils import associative_scan, binary_operator_diag
import torch.jit as jit

#MLPs

def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

class FirstChannel(nn.Module):
    def __init__(self, cout, scale=1.0):
        super().__init__()
        self.cout = cout
        self.scale = scale

    def forward(self, x):
        xdim = len(x.shape)
        if xdim == 4:
            return self.scale * x[:, :self.cout, :, :]
        elif xdim == 2:
            return self.scale * x[:, :self.cout]
        elif xdim == 3:
            return self.scale * x[:, :, :]



class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features + out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B @ x
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T)  # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x


class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features + out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B*h
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5)  # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = F.relu(x) * torch.exp(self.psi)  # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T)  # sqrt(2) A^top \Psi z
        return x

#LRUs

class LRU(nn.Module):
    """ Linear Recurrent Unit. The LRU is simulated using Parallel Scan (fast!) when
     "scan" is set to True (default) in the forward pass, otherwise recursively (slow)."""

    def __init__(
            self, in_features: int, out_features: int, state_features: int, rmin=0.9, rmax=1.0, max_phase=6.283
    ):
        super().__init__()
        self.out_features = out_features
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features)
        )
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(torch.ones_like(lambda_abs) - torch.square(lambda_abs))
            )
        )
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))  # N, U
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))  # H, N

        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)

        lambda_re = lambda_abs * torch.cos(lambda_phase)
        lambda_im = lambda_abs * torch.sin(lambda_phase)
        lambdas = torch.complex(lambda_re, lambda_im)
        # lambdas = lambda_abs*torch.exp(1j*lambda_phase)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B = gammas * self.B
        return lambdas, B, self.C, self.D

    def ss_real_matrices(self, to_numpy=True):

        lambdas, B, self.C, self.D = self.ss_params()

        lambdas_full = torch.zeros(2 * self.state_features, device=lambdas.device, dtype=lambdas.dtype)
        lambdas_full[::2] = lambdas
        lambdas_full[1::2] = lambdas.conj()

        # First convert to complex conjugate system....
        A_full = torch.diag(lambdas_full)
        B_full = torch.zeros((2 * self.state_features, self.in_features), device=lambdas.device, dtype=lambdas.dtype)
        B_full[::2] = B
        B_full[1::2] = B.conj()
        C_full = torch.zeros((self.out_features, 2 * self.state_features), device=lambdas.device, dtype=lambdas.dtype)
        C_full[:, ::2] = 0.5 * self.C  # we take the real part of the complex conjugate system as output...
        C_full[:, 1::2] = 0.5 * self.C.conj()
        D_full = self.D

        # Then apply transformation to real domain
        T_block = torch.tensor([[1, 1], [1j, -1j]], device=lambdas.device, dtype=lambdas.dtype)
        T_block_inv = torch.linalg.inv(T_block)
        T_full = torch.block_diag(*([T_block] * self.state_features))
        T_full_inv = torch.block_diag(*([T_block_inv] * self.state_features))

        A_real = (T_full @ A_full @ T_full_inv).real
        B_real = (T_full @ B_full).real
        C_real = (C_full @ T_full_inv).real
        D_real = D_full

        ss_real_params = [A_real, B_real, C_real, D_real]
        if to_numpy:
            ss_real_params = [ss_real_param.detach().numpy() for ss_real_param in ss_real_params]

        return (*ss_real_params,)

    def forward_loop(self, input, state=None):

        # Input size: (B, L, H)
        lambdas, B, C, D = self.ss_params()
        output = torch.empty(
            [i for i in input.shape[:-1]] + [self.out_features], device=self.B.device
        )

        states = []
        for u_step in input.split(1, dim=1):  # 1 is the time dimension

            u_step = u_step.squeeze(1)
            state = lambdas * state + u_step.to(B.dtype) @ B.T
            states.append(state)

        states = torch.stack(states, 1)
        output = (states @ C.mT).real + input @ D.T

        return output, states

    @torch.compiler.disable
    def forward_scan(self, input, state=None):

        # Only handles input of size (B, L, H)
        # Batched parallel scan, borrows heavily from https://colab.research.google.com/drive/1RgIv_3WAOW53CS0BnT7_782VKTYis9WG?usp=sharing
        # which in turn borrows from https://github.com/i404788/s5-pytorch
        lambdas, B, C, D = self.ss_params()

        # lambdas is shape (N,) but needs to be repeated to shape (L, N),
        # since input_sequence has shape (B, L, H).
        lambda_elements = lambdas.tile(input.shape[1], 1)
        # Calculate B@u for each step u of each input sequence in the batch.
        # Bu_elements will have shape (B, L, N)
        Bu_elements = input.to(B.dtype) @ B.T
        if state is not None:
            Bu_elements[:, 0, :] = Bu_elements[:, 0, :] + lambdas * state
            # Vmap the associative scan since Bu_elements is a batch of B sequences.
        # Recall that Lambda_elements has been repeated L times to (L, N),
        # while Bu_seq has shape (B, L, N)
        inner_state_fn = lambda Bu_seq: associative_scan(binary_operator_diag, (lambda_elements, Bu_seq))[1]
        # inner_states will be of shape (B, L, N)
        inner_states = torch.vmap(inner_state_fn)(Bu_elements)

        # y = (inner_states @ self.C.T).real + input_sequences * self.D
        y = (inner_states @ C.T).real + input @ D.T
        return y, inner_states

    def forward(self, input, gamma=None, state=None, mode="scan"):

        if state is None:
            state = torch.view_as_complex(
                torch.zeros((self.state_features, 2), device=input.device)
            )  # default initial state, size N

        match mode:
            case "scan":
                y = self.forward_scan(input, state)
                return y
            case "loop":
                y, s = self.forward_loop(input, state)
                return y, s


# WORK IN PROGRESS

class LRU_Robust(jit.ScriptModule):
    """ Implements a Linear Recurrent Unit (LRU) with trainable or prescribed l2 gain gamma.
    No parallel scan implementation available at the moment. """

    def __init__(self, state_features: int, trainable: bool):
        super().__init__()
        self.trainable = trainable
        self.state_features = state_features
        self.register_buffer('state', torch.zeros(state_features))
        self.register_buffer('ID', torch.eye(state_features))

        self.alpha = nn.Parameter(torch.tensor(-1.8)) # controls the initialization of the matrix A:
        # the more negative the alpha at initialization, the closer the eigenvalues of A will be
        # to the boundary of the unitary circle at initialization. This helps the SSM to obtain long memory properties.


        if self.trainable:
            self.gamma = nn.Parameter(10 * torch.randn(1, 1)) # l2 gain
            self.epsilon = nn.Parameter(torch.tensor([.9]))
        else:
            self.register_buffer('gamma', torch.tensor(3))
        self.Skew = nn.Parameter(torch.randn(state_features, state_features))

        # Define each block of X as a parameter
        self.X11 = nn.Parameter(torch.eye(state_features))
        self.X12 = nn.Parameter(torch.eye(state_features))
        self.X22 = nn.Parameter(torch.eye(state_features))
        self.X21 = nn.Parameter(torch.eye(state_features))

        self.C = nn.Parameter(torch.eye(state_features))
        self.D = nn.Parameter(torch.eye(state_features))




    @jit.script_method
    def set_param(self, gamma_lru = None):  # Parameter update for l2 gain (free param)

        gamma = self.gamma
        if not self.trainable and gamma_lru is not None:
            gamma = gamma_lru

        epsilon = gamma**2 * torch.sigmoid(self.alpha)

        # Create a skew-symmetric matrix
        Sk = self.Skew - self.Skew.T
        # Create orthogonal matrix via Cayley Transform
        Q = (self.ID - Sk) @ torch.linalg.inv(self.ID + Sk)

        # Compute the blocks of H= X*X.T
        HHt_22 = self.X21 @ self.X21.T + self.X22 @ self.X22.T + self.D.T @ self.D
        lmax= torch.max(torch.linalg.eigvals(HHt_22).real)
        normfactor = (gamma**2-epsilon)/lmax
        tnorm = torch.sqrt(normfactor)
        # Define the normalized blocks
        X21n = self.X21 * tnorm
        X22n = self.X22 * tnorm
        Dn = self.D * tnorm
        HHt_22n = HHt_22 * normfactor

        HHt_11 = self.X11 @ self.X11.T + self.X12 @ self.X12.T + self.C.T @ self.C

        HHt_12 = self.X11 @ X21n.T + self.X12 @ X22n.T + self.C.T @ Dn
        HHt_21 = HHt_12.T

        # # Assemble H*H.T in block form
        # HHt = torch.cat([
        #     torch.cat([HHt_11, HHt_12], dim=1),
        #     torch.cat([HHt_21, HHt_22n], dim=1)
        # ], dim=0)

        V = HHt_22n-gamma**2*self.ID
        R = HHt_12 @ torch.linalg.inv(V).T @ HHt_12.T

        CR = torch.linalg.cholesky(-R)
        CRH = torch.linalg.cholesky(-R + HHt_11)

        Atilde = CRH @ Q @ torch.linalg.inv(CR)

        A = torch.linalg.inv(Atilde).T
        #P = -Atilde @ R @ Atilde.T
        #la= torch.abs(torch.linalg.eigvals(A))
        # lp = torch.linalg.eigvals(self.P)
        B = torch.linalg.pinv(HHt_12.T @ Atilde.T) @ V.T
        C = self.C

        # row1 = torch.cat([-A.T@P@ A+P, -A.T@P@B], dim=1)
        # row2 = torch.cat([-(A.T@P@B).T, -B.T@P@B+(gamma**2*self.ID)], dim=1)
        # M = torch.cat([row1, row2], dim=0)
        # eigs = torch.linalg.eigvals(M)

       #eigs
        return A, B, C, Dn

    @jit.script_method
    def forward(self, input, gamma = None, state=None, mode: str="scan"):
        state = torch.zeros(self.state_features, device=self.C.device)
        # Input size: (B, L, H)
        A, B, C, D = self.set_param(gamma)
        if state is None:
            state = torch.zeros(self.state_features)
        output = torch.empty(
            [i for i in input.shape[:-1]] + [self.state_features], device=self.C.device
        )

        states = []
        for u_step in input.split(1, dim=1):  # 1 is the time dimension

            u_step = u_step.squeeze(1)
            state = state @ A.T + u_step @ B.T
            states.append(state)

        states = torch.stack(states, 1)
        output = states @ C.mT + input @ D.T

        return output, states

#overall SSM

@dataclass
class DWNConfig:
    d_model: int = 10 # input/output size of the LRU (u and y)
    d_state: int = 64 # state size of the LRU (n)
    n_layers: int = 6 # number of SSMs blocks in cascade for deep structures
    dropout: float = 0.0 # set it different from 0 if you want to introduce dropout regularization
    bias: bool = True # bias of MLP layers
    rmin: float = 0.0 # min. magnitude of the eigenvalues at initialization in the complex parametrization
    rmax: float = 1.0 # max. magnitude of the eigenvalues at initialization in the complex parametrization
    max_phase: float = 2 * math.pi # maximum phase of the eigenvalues at initialization in the complex parametrization
    ff: str = "MLP" # non-linear block used in the scaffolding
    scale: float = 1 # Lipschitz constant of the Lipschitz bounded MLP (LMLP)
    dim_amp: int = 4 # controls the hidden layer's dimension of the MLP
    gamma: bool = True # set this to true if you want to use the l2 gain parametrization for the SSM. If set to false,
    # the complex diagonal parametrization of the LRU will be used instead.
    gain: float = 8 # set the overall l2 gain in case you want to keep it fixed and not trainable
    trainable: bool = True # set this to true if you want a trainable l2 gain.

    # Parallel scan must be selected in the forward call. It will be disabled when gamma is set to True.


    """ Scaffolding Layers """


class MLP(nn.Module):
    """ Standard Transformer MLP """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.dim_amp * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.dim_amp * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LMLP(nn.Module):
    """ Implements a Lipschitz.-bounded MLP with sandwich layers. The square root
    # of the Lipschitz bound is given by scale """

    def __init__(self, config: DWNConfig):
        super().__init__()
        layers = [FirstChannel(config.d_model, scale=config.scale),
                  SandwichFc(config.d_model, config.dim_amp * config.d_model, bias=False, scale=config.scale),
                  SandwichFc(config.dim_amp * config.d_model, config.dim_amp * config.d_model, bias=False,
                             scale=config.scale),
                  SandwichFc(config.dim_amp * config.d_model, config.dim_amp * config.d_model, bias=False,
                             scale=config.scale),
                  SandwichLin(config.dim_amp * config.d_model, config.d_model, bias=False, scale=config.scale),
                  nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        x = self.model(input)
        return x


class GLU(nn.Module):
    """ The static non-linearity used in the S4 paper """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),
            # nn.Conv1d(config.d_model, 2 * config.d_model, kernel_size=1),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        x = self.output_linear(x)
        return x


    """ SSMs blocks """


class DWNBlock(nn.Module):
    """ SSM block: LRU --> MLP + skip connection """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        if config.gamma:
            self.lru = LRU_Robust(config.d_model, config.trainable)

        else:
            self.lru = LRU(config.d_model, config.d_model, config.d_state,
                           rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)
        match config.ff:
            case "GLU":
                self.ff = GLU(config)
            case "MLP":
                self.ff = MLP(config)
            case "LMLP":
                self.ff = LMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, gamma=None, state=None, mode: str ="scan"):

        z = x
        #  z = self.ln(z)  # prenorm

        z, state = self.lru(z, gamma, state, mode)

        z = self.ff(z)  # MLP, GLU or LMLP
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x, state


class DWN(nn.Module):
    """ Deep SSMs block: encoder --> cascade of n SSMs --> decoder  """

    def __init__(self, n_u: int, n_y: int, config: DWNConfig):
        super().__init__()

        self.config = config

        self.encoder = nn.Linear(n_u, config.d_model, bias=False)
        self.decoder = nn.Linear(config.d_model, n_y, bias=False)

        if not config.trainable: # parameters needed for when the l2 gain is fixed and prescribed
            self.alpha = nn.Parameter(torch.randn(1))
            self.gamma_e = nn.Parameter(torch.randn(1))
            self.register_buffer('gamma_t', torch.tensor(config.gain))
            self.encoder = nn.Parameter(torch.randn(n_u, config.d_model))
            self.decoder = nn.Parameter(torch.randn(config.d_model, n_y))




        self.blocks = nn.ModuleList([DWNBlock(config) for _ in range(config.n_layers)])

    def forward_fixed_gamma(self, u, state=None, mode="scan"):

        gamma_t = torch.abs(self.gamma_t)
        gamma_e = torch.abs(self.gamma_e)
        gamma_mid = torch.pow(1 / torch.sigmoid(self.alpha), 1 / self.config.n_layers) * torch.ones(
            self.config.n_layers)
        gamma_d = torch.sigmoid(self.alpha) * gamma_t / self.gamma_e
        encoder = gamma_e * self.encoder / torch.norm(self.encoder, 2)
        decoder = gamma_d * self.decoder / torch.norm(self.decoder, 2)
        x = u@encoder
        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x, state_block = block(x, gamma_mid[layer]-1, state=state_block, mode=mode)
        x = x@decoder

        return x, state

    def forward_trainable_gamma(self, u, state=None, mode="scan"):

        x = self.encoder(u)
        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x, state_block = block(x, state=state_block, mode=mode)
        x = self.decoder(x)

        return x, state


    def forward(self, u, state=None, mode="scan"):

        if not self.config.trainable:
            x, state = self.forward_fixed_gamma(u, state, mode)
        else:
            x, state = self.forward_trainable_gamma(u, state, mode)

        return x, state