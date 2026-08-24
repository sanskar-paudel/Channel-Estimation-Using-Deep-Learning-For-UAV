#import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(42)
np.random.seed(42)

NUM_SAMPLES = 100_000
L           = 10
NUM_BLOCKS  = NUM_SAMPLES // L

TRAIN_RATIO = 0.8
BOTTLENECK  = 2          
HIDDEN      = 128
EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 1e-3
LAMBDA_CH   = 1.0        # weight for channel supervision loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


SNR_dB    = np.random.randint(0, 21, size=NUM_SAMPLES).astype(float)


x_real = 2 * np.random.randint(0, 2, NUM_SAMPLES) - 1
x_imag = 2 * np.random.randint(0, 2, NUM_SAMPLES) - 1
x      = (x_real + 1j * x_imag) / np.sqrt(2)


h = (np.random.randn(NUM_SAMPLES) + 1j * np.random.randn(NUM_SAMPLES)) / np.sqrt(2)

noise_std = 10 ** (-SNR_dB / 20)
noise     = (noise_std * (np.random.randn(NUM_SAMPLES) +
             1j * np.random.randn(NUM_SAMPLES)) / np.sqrt(2))

y = h * x + noise

n_valid  = NUM_BLOCKS * L
Y_real   = np.real(y[:n_valid]).reshape(NUM_BLOCKS, L)
Y_imag   = np.imag(y[:n_valid]).reshape(NUM_BLOCKS, L)
X_data   = np.hstack([Y_real, Y_imag]).astype(np.float32)  

# True per-block channel label - shape (NUM_BLOCKS, 2)
h_blocks     = h[:n_valid].reshape(NUM_BLOCKS, L).mean(axis=1)
H_label      = np.stack([np.real(h_blocks),
                          np.imag(h_blocks)], axis=1).astype(np.float32)

train_size = int(TRAIN_RATIO * NUM_BLOCKS)

X_train = torch.tensor(X_data[:train_size])
X_test  = torch.tensor(X_data[train_size:])
H_train = torch.tensor(H_label[:train_size])
H_test  = torch.tensor(H_label[train_size:])

train_ds     = TensorDataset(X_train, H_train)
test_ds      = TensorDataset(X_test,  H_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

class ChannelAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden, latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat          # return both bottleneck AND reconstruction

model     = ChannelAutoencoder(2*L, HIDDEN, BOTTLENECK).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}\n")

train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # ── train ──
    model.train()
    running = 0.0
    for xb, hb in train_loader:
        xb, hb = xb.to(DEVICE), hb.to(DEVICE)
        optimizer.zero_grad()
        z, x_hat = model(xb)
        loss_recon = criterion(x_hat, xb)      # reconstruction
        loss_ch    = criterion(z, hb)          # channel estimation (supervised)
        loss       = loss_recon + LAMBDA_CH * loss_ch
        loss.backward()
        optimizer.step()
        running += loss_ch.item() * xb.size(0)   # track channel MSE

    train_losses.append(running / len(train_loader.dataset))
    scheduler.step()

    
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for xb, hb in test_loader:
            xb, hb = xb.to(DEVICE), hb.to(DEVICE)
            z, _   = model(xb)
            val_running += criterion(z, hb).item() * xb.size(0)

    val_losses.append(val_running / len(test_loader.dataset))

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train CH-MSE: {train_losses[-1]:.6f} | "
              f"Val CH-MSE:   {val_losses[-1]:.6f}")

# Channel Estimation
model.eval()
with torch.no_grad():
    latent, _ = model(X_test.to(DEVICE))
    latent    = latent.cpu().numpy()

h_estimated = latent[:, 0] + 1j * latent[:, 1]
h_true      = H_test[:, 0].numpy() + 1j * H_test[:, 1].numpy()

mse = np.mean(np.abs(h_estimated - h_true) ** 2)
print(f"\n{'='*52}")
print(f"  Pilot-Free Channel Estimation MSE = {mse:.6f}")
print(f"{'='*52}\n")

def moving_avg(arr, w=10):
    return np.convolve(arr, np.ones(w) / w, mode="valid")

SMOOTH_W = 10

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle("Autoencoder-Based Channel Estimation (UAV)",
             fontsize=14, fontweight="bold")

ax = axes[0]
ax.semilogy(moving_avg(train_losses, SMOOTH_W), "b", linewidth=2, label="Train loss")
ax.semilogy(moving_avg(val_losses,   SMOOTH_W), "r", linewidth=2, label="Val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Channel MSE (log scale)")
ax.set_title("Smoothed Training vs Validation Loss")
ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)

N_PLOT = 100
ax = axes[1]

h_true_merged       = np.empty(2 * N_PLOT)
h_true_merged[0::2] = np.real(h_true[:N_PLOT])
h_true_merged[1::2] = np.imag(h_true[:N_PLOT])

h_est_merged        = np.empty(2 * N_PLOT)
h_est_merged[0::2]  = np.real(h_estimated[:N_PLOT])
h_est_merged[1::2]  = np.imag(h_estimated[:N_PLOT])

ax.plot(h_true_merged, "b",   linewidth=1.5, label="True Channel")
ax.plot(h_est_merged,  "r--", linewidth=1.5, label="Estimated Channel")
ax.set_xlabel("Sample Index ") #[even = Re,  odd = Im]
ax.set_ylabel("Channel Value")
ax.set_title("True vs Estimated Channel")
ax.legend()
ax.grid(True, ls="--", alpha=0.5)

plt.tight_layout()
#save_path = os.path.join(SCRIPT_DIR, "uav_channel_estimation_results.png")
#plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
#print(f"Figure saved to: {save_path}")
