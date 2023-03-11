import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# TODO:
# trainable sigma
# mlp instead of residual blocks

device = torch.device("cuda")

input_dimension = 2
n_steps = 100  # number of denoising time steps


def get_betas(max_beta=0.2):

    def a_f(t):
        return torch.cos((t / n_steps + 0.008) / 1.008 * 0.5 * torch.pi)**2

    ts = torch.linspace(0, n_steps - 1, n_steps, device=device, dtype=torch.float32)
    return torch.min(1 - a_f(ts + 1) / a_f(ts), torch.tensor(max_beta)).view(-1, 1)


# betas = torch.linspace(0.001, 0.1, n_steps, device=device).view(-1, 1)
betas = get_betas()
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
eps_k_square = ((1 - alphas)**2) / (1 - alphas_cumprod)
eps_k = torch.sqrt(eps_k_square)

alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (0, 0, 1, 0), value=1.0)
posterior_variance = (1 - alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def q_sample(x_0, t, noise):
    """
    Sample x at time t given the value of x at t=0 and the noise
    """
    return sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise


if False:
    # plt.plot(sqrt_alphas_cumprod.cpu(), label="sqrt_alphas_cumprod")
    # plt.plot(sqrt_one_minus_alphas_cumprod.cpu(), label="sqrt_one_minus_alphas_cumprod")
    plt.plot(posterior_variance.cpu(), label="posterior_variance")
    plt.show()

    theta_0 = torch.rand((1024), device=device) * 2 * torch.pi
    # x0 = torch.stack([torch.cos(theta_0), torch.sin(theta_0)]).permute(1, 0)
    x0 = torch.stack([theta_0 / torch.pi - 1, torch.sin(theta_0)]).permute(1, 0)

    plt.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), s=1)
    plt.show()

    for t in [1, n_steps // 10, n_steps // 2, n_steps - 1]:
        noised_x = q_sample(x0, torch.tensor(t, device=device), torch.randn((1024, 2), device=device))
        plt.scatter(noised_x[:, 0].cpu().numpy(), noised_x[:, 1].cpu().numpy(), s=1)
        # plt.hist(noised_x.cpu().numpy(), bins=100, alpha=0.5, label=f"t={t}")
    plt.show()


def sinusoid_positional_encoding(length, dimensions):

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (i // 2) / dimensions) for i in range(dimensions)]

    PE = torch.tensor([get_position_angle_vec(i) for i in range(length)], device=device)
    PE[:, 0::2] = torch.sin(PE[:, 0::2])  # dim 2i
    PE[:, 1::2] = torch.cos(PE[:, 1::2])  # dim 2i+1
    return PE.to(dtype=torch.float32)


class ResidualBlock(torch.nn.Module):

    def __init__(self, x_dim):
        super(ResidualBlock, self).__init__()
        hid_dim = 32
        # self.pe_dim = 8
        # final_dim = 2 * 2 * self.pe_dim * x_dim
        # self.pe_dim_coeff = (2**torch.linspace(0, self.pe_dim - 1, self.pe_dim)).to(device)
        self.pe = sinusoid_positional_encoding(n_steps, hid_dim)
        self.linear0 = torch.nn.Linear(x_dim * 2, hid_dim)
        self.linear1 = torch.nn.Linear(hid_dim, hid_dim)
        self.linear2 = torch.nn.Linear(hid_dim, x_dim * 2)
        self.non_linear = torch.nn.ReLU()
        self.output_non_linear = torch.nn.Tanh()

    def forward(self, tup):
        x, t = tup
        residual = x
        # x = (self.pe_dim_coeff * x[..., None]).view(-1, 2 * self.pe_dim * input_dimension)
        # x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        # x = x + self.pe[t]
        x = self.non_linear(self.linear0(x)) + self.pe[t]
        x = self.non_linear(self.linear1(x))
        x = self.output_non_linear(self.linear2(x)) + residual
        return x, t


class DenoiseModel(torch.nn.Module):

    def __init__(self, x_dim):
        super(DenoiseModel, self).__init__()
        layers = []
        for i in range(10):
            layers.append(ResidualBlock(x_dim))
        self.layers = torch.nn.Sequential(*layers)
        self.sigma_non_linear = torch.nn.Sigmoid()

    def forward(self, x, t):
        x, t = self.layers((x.repeat(1, 2), t))
        sigma_v = self.sigma_non_linear(x[..., input_dimension:])
        x = x[..., :input_dimension]
        sigma_inv = torch.exp(-sigma_v * torch.log(betas[t]) - (1 - sigma_v) * torch.log(posterior_variance[t]))
        return x, sigma_inv


denoiser = DenoiseModel(input_dimension).to(device=device)


def sample_loss(x, t, weight=1):
    noise = torch.randn(x.shape, device=device)
    noisy_x = q_sample(x, t, noise)
    noise_computed, sigma_inv = denoiser(noisy_x, t)
    noise = noise * weight
    noise_computed = noise_computed * weight
    lamb = 0.1
    L_simple = F.mse_loss(noise, noise_computed)
    extra = sigma_inv * eps_k_square[t] / (2 * alphas[t])
    noise = noise * extra
    noise_computed = noise_computed * extra
    L_vlb = F.mse_loss(noise, noise_computed)
    return L_simple + lamb * L_vlb


@torch.no_grad()
def target_sampling_weight(x):
    """
    Target sampling weight that DDPM learns from
    """
    r = torch.norm(x, dim=1)
    w = torch.exp(-(r - 1.0)**2 / 0.05)
    return w.view(-1, 1)
    # return torch.pow(torch.sin(5 * x[..., 0]), 2).view(-1, 1)


@torch.no_grad()
def target_sampling_weight_grad_log(x):
    """
    Target sampling weight that DDPM learns from
    """
    r = torch.norm(x, dim=1).view(-1, 1)
    return -2 * (r - 1.0) * x / (0.05 * r)
    # r = torch.zeros_like(x)
    # r[..., 0] = 10 * torch.cos(5 * x[..., 0]) * torch.sin(5 * x[..., 0])
    # return r


@torch.no_grad()
def p_sample(x, t, grad_scale=0.01, print_sigma=False):
    """
    One step of revese process sampling - Algorithm 2 from the paper
    """
    alpha_t = alphas[t]
    eps_k_t = eps_k[t]
    eps, sigma_inv = denoiser(x, t)
    model_mean = grad_scale * target_sampling_weight_grad_log(x) + torch.sqrt(1.0 / alpha_t) * (x - eps_k_t * eps)
    # if print_sigma:
    #     print(torch.rsqrt(sigma_inv))
    #     print(torch.sqrt(betas[t]), torch.sqrt(posterior_variance[t]))
    return model_mean + torch.rsqrt(sigma_inv) * torch.randn(x.shape, device=device)


denoiseOptimizer = torch.optim.Adam(denoiser.parameters())


@torch.no_grad()
def get_sample(batch_size, print_sigma=False):
    x = torch.randn((batch_size, input_dimension), device=device, requires_grad=True)
    return x
    for t in range(n_steps - 1, 0, -1):
        x = p_sample(x, torch.tensor(t, device=device).view(-1), print_sigma=print_sigma)
    return x


n_epochs = 5000
batch_size = 16384
for step in range(n_epochs):
    denoiseOptimizer.zero_grad()
    t = torch.randint(1, n_steps, (batch_size, ), device=device)  # Pick random time step
    theta = torch.rand((batch_size), device=device) * 2 * torch.pi
    # x = torch.randint(1, 3, (batch_size, 1), device=device) * torch.stack([torch.cos(theta),
    #                                                                        torch.sin(theta)]).permute(1, 0)
    # x = torch.stack([theta / torch.pi - 1, torch.sin(5 * theta)]).permute(1, 0)
    x = get_sample(batch_size)
    # weight = target_sampling_weight(x)
    # weight_grad = target_sampling_weight_grad_log(x)
    s_loss = sample_loss(x, t)
    s_loss.backward()
    if step % (n_epochs // 10) == 0:
        print(f"s_loss={s_loss.item():.4f}")
    denoiseOptimizer.step()
print(f"final: s_loss={s_loss.item():.4f}")

fig_num = 5
fig_delta = n_steps // fig_num
fig, axs = plt.subplots(nrows=1, ncols=fig_num + 1, figsize=(25, 4))
for _ in range(1):
    x_gen = torch.randn((4096, input_dimension), device=device)
    # x_gen = torch.rand((2048, input_dimension), device=device) * 2 - 1
    axs[5].scatter(x_gen[:, 0].cpu().numpy(), x_gen[:, 1].cpu().numpy(), s=1)
    for i in range(n_steps - 1, 0, -1):  # 49->1
        # x_gen = p_sample(x_gen, torch.tensor(i, device=device).view(-1), 0)
        x_gen = p_sample(x_gen, torch.tensor(i, device=device).view(-1))
        if i % fig_delta == 0:
            axs[i // fig_delta].scatter(x_gen[:, 0].cpu().numpy(), x_gen[:, 1].cpu().numpy(), s=1)
    axs[0].scatter(x_gen[:, 0].cpu().numpy(), x_gen[:, 1].cpu().numpy(), s=1)
    plt.show()
# plt.hlines(x0, 0, n_steps, color="black", linestyle="--", label=f"x0 = {x0}")
# plt.hlines(x1, 0, n_steps, color="black", linestyle="--", label=f"x1 = {x1}")
# plt.show()
# hard case
# +physics
# + prior