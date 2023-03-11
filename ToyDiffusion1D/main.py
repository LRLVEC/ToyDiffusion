import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

device = torch.device("cuda")
input_dimension = 1
n_steps = 50  # number of denoising time steps

x0 = torch.tensor([-2]).view(1, input_dimension)  # "realistic" sample which the model needs to learn to generate
x1 = torch.tensor([2]).view(1, input_dimension)

alphas = 1. - torch.linspace(0.001, 0.2, n_steps, device=device)
alphas_cumprod = torch.cumprod(alphas, axis=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = (1 - alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def q_sample(x_0, t, noise):
    """
    Sample x at time t given the value of x at t=0 and the noise
    """
    t = t.view(-1)
    return sqrt_alphas_cumprod.gather(-1, t).view(-1, 1) * x_0 + sqrt_one_minus_alphas_cumprod.gather(
        -1, t).view(-1, 1) * noise


if False:
    plt.plot(sqrt_alphas_cumprod, label="sqrt_alphas_cumprod")
    plt.plot(sqrt_one_minus_alphas_cumprod, label="sqrt_one_minus_alphas_cumprod")
    plt.plot(posterior_variance, label="posterior_variance")
    plt.show()

    for t in [1, n_steps // 10, n_steps // 2, n_steps - 1]:
        noised_x = q_sample(x0, torch.tensor(t), torch.randn(1000).view(-1, 1))
        plt.hist(noised_x.numpy(), bins=100, alpha=0.5, label=f"t={t}")
    plt.show()

    res = [(t, q_sample(x0, torch.tensor(t), torch.randn(1)).item()) for _ in range(10) for t in range(n_steps)]
    x, y = list(zip(*res))
    plt.scatter(x, y, s=1)
    plt.xlabel("time")
    plt.ylabel("x")
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
        self.pe = sinusoid_positional_encoding(n_steps, hid_dim)
        self.linear0 = torch.nn.Linear(x_dim, hid_dim)
        self.linear1 = torch.nn.Linear(hid_dim, hid_dim)
        self.linear2 = torch.nn.Linear(hid_dim, x_dim)
        self.non_linear = torch.nn.ReLU()
        self.output_non_linear = torch.nn.Tanh()

    def forward(self, tup):
        x, t = tup
        residual = x
        x = self.non_linear(self.linear0(x)) + self.pe[t.view(-1)]
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

    def forward(self, x, t):
        x, t = self.layers((x, t))
        return x


denoise = DenoiseModel(input_dimension).to(device=device)


def p_loss(x, t):
    noise = torch.randn(t.shape, device=device)  # Generate a noise
    noisy_x = q_sample(x, t, noise)  # Compute x at time t with this value of the noise - forward process
    noise_computed = denoise(noisy_x, t)  # Use our trained model to predict the value of the noise, given x(t) and t
    return F.mse_loss(noise, noise_computed)  # Compare predicted value of the noise with the actual value


optimizer = torch.optim.Adam(denoise.parameters())

n_epochs = 5000
batch_size = 1024
for step in range(n_epochs):
    optimizer.zero_grad()
    t = torch.randint(0, n_steps, (batch_size, 1), device=device)  # Pick random time step
    x = torch.randint(-4, 5, (batch_size, 1), device=device).to(dtype=torch.float32)
    loss = p_loss(x, t)
    loss.backward()
    if step % (n_epochs // 10) == 0:
        print(f"loss={loss.item():.4f}")
    optimizer.step()
print(f"final: loss={loss.item():.4f}")


def p_sample(x, t):
    """
    One step of revese process sampling - Algorithm 2 from the paper
    """
    alpha_t = alphas.gather(-1, t)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.gather(-1, t)
    # Get mean x[t - 1] conditioned at x[t] - see eq. (11) in the paper
    model_mean = torch.sqrt(
        1.0 / alpha_t) * (x - (1 - alpha_t) * denoise(x, t.view(-1, 1)) / sqrt_one_minus_alphas_cumprod_t)
    # Get variance of x[t - 1]
    model_var = posterior_variance.gather(-1, t)
    # Samples for the normal distribution with given mean and variance
    return model_mean + torch.sqrt(model_var) * torch.randn(1, device=device)


plt.figure(figsize=(10, 10))
for _ in range(50):
    x_gens = []
    x_gen = torch.randn((1, 1), device=device)
    for i in range(n_steps - 1, 0, -1):
        x_gen = p_sample(x_gen, torch.tensor(i, device=device).view(-1))
        x_gens.append(x_gen.cpu().detach().numpy()[0])
    plt.plot(x_gens[::-1])
plt.hlines(x0, 0, n_steps, color="black", linestyle="--", label=f"x0 = {x0}")
plt.hlines(x1, 0, n_steps, color="black", linestyle="--", label=f"x1 = {x1}")
plt.show()
