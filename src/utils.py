import torch
import matplotlib.pyplot as plt


def normalize(x, xmin, xmax):
    """Normalize to [-1, 1]."""
    return 2 * (x - xmin) / (xmax - xmin) - 1

def plot_slices(model, device, times, Xmin, Xmax, Ymin, Ymax, Nx=32, Ny=32):
    """Plot grids of [Xmin, Xmax] x [Ymin, Ymin] for each item in the list times."""
    model.eval()
    x = torch.linspace(Xmin, Xmax, Nx)
    y = torch.linspace(Ymin, Ymax, Ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    for t_val in times:
        t_tensor = torch.full((coords.shape[0], 1), t_val, device=device)
        input = torch.cat([coords, t_tensor], dim=1)
        with torch.no_grad():
            pred = model(input)
        c1 = pred[:, 0].reshape(Nx, Ny).cpu()
        c2 = pred[:, 1].reshape(Nx, Ny).cpu()
        c3 = 1 - c1 - c2

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for i, (ci, name) in enumerate(zip([c1, c2, c3], ['c1', 'c2', 'c3'])):
            im = axs[i].imshow(ci.T, extent=[Xmin, Xmax, Ymin, Ymax], origin='lower', cmap='Greys')
            axs[i].set_title(f"{name} at t = {t_val:.2f}")
            plt.colorbar(im, ax=axs[i])
        plt.tight_layout()
        plt.show()