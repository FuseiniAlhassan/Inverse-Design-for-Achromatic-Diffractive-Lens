import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Create results directory
os.makedirs('results', exist_ok=True)

# ----------------------------
# 1. Simulation Parameters
# ----------------------------

def setup_parameters():
    """Set up simulation parameters and grids."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = 256  # grid size
    L = 5e-3  # physical size (5 mm)
    lambda_wl = torch.tensor([450e-9, 500e-9, 550e-9, 600e-9, 650e-9], device=device)  # More wavelengths for better achromaticity
    focal_length = 0.02  # 2 cm

    # Coordinate grid
    x = torch.linspace(-L/2, L/2, N, device=device)
    X, Y = torch.meshgrid(x, x, indexing='xy')
    r2 = X**2 + Y**2

    # Gaussian target focal spot (more realistic than delta; sigma=1 pixel)
    focus_target = torch.exp(-((X)**2 + (Y)**2) / (2 * (L/N)**2)).to(device)
    focus_target /= focus_target.max()  # Normalize to 1

    return device, N, L, lambda_wl, focal_length, X, Y, focus_target

device, N, L, lambda_rgb, focal_length, X, Y, focus_target = setup_parameters()

# ----------------------------
# 2. Angular Spectrum Propagation
# ----------------------------

def angular_spectrum_propagation(field, wavelength, z):
    """Propagate the field using angular spectrum method."""
    k = 2 * np.pi / wavelength
    fx = torch.fft.fftfreq(N, d=L/N).to(device)
    FX, FY = torch.meshgrid(fx, fx, indexing='xy')
    arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    arg = torch.clamp(arg, min=0)  # Avoid NaNs in sqrt
    H = torch.exp(1j * z * k * torch.sqrt(arg))
    return torch.fft.ifft2(torch.fft.fft2(field) * H)

# ----------------------------
# 3. Phase Mask Model and Optimizer
# ----------------------------

def setup_model_and_optimizer():
    """Initialize phase mask and optimizer with scheduler."""
    phase_mask = torch.nn.Parameter(torch.rand((N, N), device=device) * 2 * np.pi)
    optimizer = torch.optim.Adam([phase_mask], lr=0.03)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    return phase_mask, optimizer, scheduler

phase_mask, optimizer, scheduler = setup_model_and_optimizer()

# ----------------------------
# 4. Optimization
# ----------------------------

def total_variation_loss(phase):
    """Regularization: Total variation for phase smoothness."""
    dx = torch.abs(phase[1:, :] - phase[:-1, :]).mean()
    dy = torch.abs(phase[:, 1:] - phase[:, :-1]).mean()
    return dx + dy

def generate_field(phase):
    """Generate complex field from phase."""
    return torch.exp(1j * phase)

def optimize_phase(num_steps=250, reg_weight=0.001):
    """Run optimization loop with regularization and scheduler."""
    loss_history = []
    frames = []

    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()
        total_loss = 0

        for wl in lambda_rgb:
            field_in = generate_field(phase_mask)
            field_out = angular_spectrum_propagation(field_in, wl, focal_length)
            intensity = torch.abs(field_out)**2
            mse_loss = torch.mean((intensity - focus_target)**2)
            total_loss += mse_loss

        # Add regularization
        reg_loss = reg_weight * total_variation_loss(phase_mask)
        total_loss += reg_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        with torch.no_grad():
            phase_mask[:] = (phase_mask + np.pi) % (2 * np.pi) - np.pi

        loss_history.append(total_loss.item())

        if step % 10 == 0:
            frames.append(phase_mask.detach().cpu().numpy())

    return loss_history, frames

loss_history, frames = optimize_phase()

# Save data
np.save('results/final_phase.npy', phase_mask.detach().cpu().numpy())
np.save('results/loss_history.npy', np.array(loss_history))

# ----------------------------
# 5. Results Visualization
# ----------------------------

def plot_phase_mask(phase):
    """Plot and save 2D phase mask."""
    plt.figure(figsize=(6, 5))
    plt.imshow(phase, cmap='twilight', extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3])
    plt.title('Optimized Phase Mask')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.colorbar(label='Phase (rad)')
    plt.tight_layout()
    plt.savefig('results/final_phase_mask.png', dpi=300)
    plt.close()

final_phase = phase_mask.detach().cpu().numpy()
plot_phase_mask(final_phase)

# Loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Optimization Convergence')
plt.grid(True)
plt.savefig('results/loss_curve.png', dpi=300)
plt.close()

# ----------------------------
# 6. 3D Phase Surface Plot
# ----------------------------

from mpl_toolkits.mplot3d import Axes3D

def plot_3d_phase(X, Y, phase):
    """Plot and save 3D surface of phase mask."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.cpu().numpy()*1e3, Y.cpu().numpy()*1e3, phase, cmap='twilight', linewidth=0, antialiased=False)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('Phase (rad)')
    ax.set_title('3D Surface of Optimized Phase Mask')
    plt.tight_layout()
    plt.savefig('results/phase_surface_3D.png', dpi=300)
    plt.close()

plot_3d_phase(X, Y, final_phase)

# ----------------------------
# 7. Animation of Optimization Progress
# ----------------------------

def create_animation(frames):
    """Create and save GIF of phase evolution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ims = []
    for frame in frames:
        im = ax.imshow(frame, animated=True, cmap='twilight', extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3])
        ims.append([im])

    ani = FuncAnimation(fig, lambda i: ims[i], frames=len(ims), interval=200, blit=True)
    ani.save('results/phase_evolution.gif', writer='pillow')
    plt.close()

create_animation(frames)

# ----------------------------
# 8. Verification: Focal Plane Intensities
# ----------------------------

def verify_focusing():
    """Simulate and plot focal intensities for each wavelength."""
    fig, axs = plt.subplots(1, len(lambda_rgb), figsize=(15, 3))
    for i, wl in enumerate(lambda_rgb):
        field_in = generate_field(phase_mask)
        field_out = angular_spectrum_propagation(field_in, wl, focal_length)
        intensity = (torch.abs(field_out)**2).detach().cpu().numpy()
        axs[i].imshow(intensity, cmap='hot', extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3])
        axs[i].set_title(f'λ = {wl.item()*1e9:.0f} nm')
        axs[i].set_xlabel('x (mm)')
        axs[i].set_ylabel('y (mm)')
    plt.tight_layout()
    plt.savefig('results/focal_intensities.png', dpi=300)
    plt.close()

verify_focusing()

# ----------------------------
# 9. Summary Output
# ----------------------------

print("Optimization complete. Results saved in /results directory:")
print("- final_phase_mask.png")
print("- loss_curve.png")
print("- phase_surface_3D.png")
print("- phase_evolution.gif")
print("- focal_intensities.png")
print("- final_phase.npy")
print("- loss_history.npy")