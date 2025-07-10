import torch

def gradients(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

def chemical_potentials(c1, c2, coords, gamma):
    c3 = 1 - c1 - c2
    grads_c1 = gradients(c1, coords)
    grads_c2 = gradients(c2, coords)
    grads_c3 = gradients(c3, coords)

    lap_c1 = gradients(grads_c1[:, 0:1], coords)[:, 0:1] + gradients(grads_c1[:, 1:2], coords)[:, 1:2]
    lap_c2 = gradients(grads_c2[:, 0:1], coords)[:, 0:1] + gradients(grads_c2[:, 1:2], coords)[:, 1:2]
    lap_c3 = gradients(grads_c3[:, 0:1], coords)[:, 0:1] + gradients(grads_c3[:, 1:2], coords)[:, 1:2]

    mu1 = 2 * c1 * (1 - c1) * (1 - 2 * c1) - gamma * lap_c1
    mu2 = 2 * c2 * (1 - c2) * (1 - 2 * c2) - gamma * lap_c2
    mu3 = 2 * c3 * (1 - c3) * (1 - 2 * c3) - gamma * lap_c3
    return mu1, mu2, mu3