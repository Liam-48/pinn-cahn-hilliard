import torch
from src.utils import normalize
from src.physics import *

def pinn_loss(model, coords, t, Xmin, Xmax, Ymin, Ymax, Tmin, Tmax,
                M, coords_0, c1_0, c2_0, gamma, coords_per_bc, t_per_bc):
    coords.requires_grad_(True)
    t.requires_grad_(True)

    # Normalization of inputs
    x_scaled = normalize(coords[:, 0:1], Xmin, Xmax)
    y_scaled = normalize(coords[:, 1:2], Ymin, Ymax)
    t_scaled = normalize(t, Tmin, Tmax)

    input_scaled = torch.cat([x_scaled, y_scaled, t_scaled], dim=1)
    output = model(input_scaled)
    c1, c2 = output[:, 0:1], output[:, 1:2]

    # Physics computations
    c1_t = gradients(c1, t)
    c2_t = gradients(c2, t)

    mu1, mu2, mu3 = chemical_potentials(c1, c2, coords, gamma)
    lap_mu = [gradients(gradients(m, coords)[:, 0:1], coords)[:, 0:1] +
              gradients(gradients(m, coords)[:, 1:2], coords)[:, 1:2]
              for m in [mu1, mu2, mu3]]
    
    # PDE loss
    rhs1 = sum(M[0, j] * lap_mu[j] for j in range(3))
    rhs2 = sum(M[1, j] * lap_mu[j] for j in range(3))
    loss_pde = ((c1_t - rhs1)**2).mean() + ((c2_t - rhs2)**2).mean()

    # IC loss
    pred_0 = model(torch.cat([coords_0, torch.zeros_like(coords_0[:, :1])], dim=1))
    loss_ic = ((pred_0[:, 0:1] - c1_0)**2).mean() + ((pred_0[:, 1:2] - c2_0)**2).mean()
    x0_scaled = normalize(coords_0[:, 0:1], Xmin, Xmax)
    y0_scaled = normalize(coords_0[:, 1:2], Ymin, Ymax)
    t0_scaled = torch.full_like(x0_scaled, -1.0)

    input_ic_scaled = torch.cat([x0_scaled, y0_scaled, t0_scaled], dim=1)
    pred_0 = model(input_ic_scaled)

    # Periodic BC loss
    x_per_scaled = normalize(coords_per_bc[:, 0:1], Xmin, Xmax)
    y_per_scaled = normalize(coords_per_bc[:, 1:2], Ymin, Ymax)
    t_per_scaled = normalize(t_per_bc, Tmin, Tmax)

    input_per_scaled = torch.cat([x_per_scaled, y_per_scaled, t_per_scaled], dim=1)
    pred_per = model(input_per_scaled)
    nx = pred_per.shape[0] // 4
    loss_bc = (
        (pred_per[0:nx] - pred_per[nx:2*nx])**2 +
        (pred_per[2*nx:3*nx] - pred_per[3*nx:4*nx])**2
    ).mean()

    return 20.0*loss_pde + 40.0*loss_ic + 5.0*loss_bc