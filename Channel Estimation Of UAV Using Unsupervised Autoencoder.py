import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42) #for reproducibility of generated data

NUM_SAMPLES = 100000
L           = 10 #no of symbols per block
NUM_BLOCKS  = int(NUM_SAMPLES / L) #total number of blocks required

TRAIN_RATIO = 0.8
BOTTLENECK  = 2    #latent dimension   
HIDDEN      = 256
EPOCHS      = 150
BATCH_SIZE  = 256
LR          = 1e-3
#DEVICE = torch.device("cpu")
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {DEVICE}")

SNR_dB = np.random.randint(0, 21, size=NUM_SAMPLES).astype(float)

x_real = 2 * np.random.randint(0, 2, NUM_SAMPLES) - 1
x_imag = 2 * np.random.randint(0, 2, NUM_SAMPLES) - 1
x      = (x_real + 1j * x_imag) / np.sqrt(2)

#h = (np.random.randn(NUM_SAMPLES) + 1j * np.random.randn(NUM_SAMPLES)) / np.sqrt(2)
# One Rayleigh channel per block
#h_block = (np.random.randn(NUM_BLOCKS) + 1j * np.random.randn(NUM_BLOCKS)) / np.sqrt(2)

# Repeat the same channel L times inside each block
#h = np.repeat(h_block, L)

# UAV Mobility Model
# Ground user position (fixed)
ground_x = 0.0
ground_y = 0.0

# UAV initial position
uav_x = 0.0
uav_y = 0.0
uav_altitude = 100.0      # meters

# UAV motion
velocity = 5.0           # m/s
dt = 1.0                  # seconds per block

# Doppler Parameters

fc = 2.4e9            # Carrier frequency (Hz)
c = 3e8               # Speed of light (m/s)

wavelength = c / fc

theta = 0             # UAV moving directly towards receiver
fd = (velocity / wavelength) * np.cos(theta)

print(f"Doppler Frequency = {fd:.2f} Hz")

# Store distance for each block
distance = np.zeros(NUM_BLOCKS)

for i in range(NUM_BLOCKS):

    # UAV moves along x-direction
    uav_x = uav_x + velocity * dt

    # Distance between UAV and ground user
    distance[i] = np.sqrt((uav_x - ground_x)**2 + (uav_y - ground_y)**2 + (uav_altitude)**2)
    
# Distance-dependent Path Loss


path_loss_exp = 2.0      # LoS environment
d0 = 1.0                 # Reference distance


# Distance-dependent Path Loss (Normalized)



# Relative path gain
path_gain = (distance[0] / distance) ** path_loss_exp
# Rician UAV Channel


K = 10                  # Rician K-factor

h_los = np.ones(NUM_BLOCKS, dtype=complex)

h_nlos = (
    np.random.randn(NUM_BLOCKS) + 1j*np.random.randn(NUM_BLOCKS)) / np.sqrt(2)

h_block = (np.sqrt(K/(K+1))*h_los + np.sqrt(1/(K+1))*h_nlos)

# Apply path loss
h_block = path_gain * h_block
print("\n===== UAV Channel Statistics =====")
print("First 10 |h_block|:", np.abs(h_block[:10]))
print("Max |h_block|:", np.max(np.abs(h_block)))
print("Min |h_block|:", np.min(np.abs(h_block)))



# Doppler Phase Rotation


time = np.arange(NUM_BLOCKS) * dt

doppler_phase = np.exp(1j * 2 * np.pi * fd * time)

h_block = h_block * doppler_phase



# Repeat for every symbol in each block
h = np.repeat(h_block, L)
noise_std = 10 ** (-SNR_dB / 20)
noise     = (noise_std * (np.random.randn(NUM_SAMPLES) + 1j * np.random.randn(NUM_SAMPLES)) / np.sqrt(2))

y = h * x + noise

n_valid = NUM_BLOCKS * L 
#n_valid = int(NUM_SAMPLES / L) * L## What is the need of n_valid? Isn't n_valid = NUM_SAMPLES?It is used to ensure that the number of samples used for training and testing is a multiple of L, which is the number of symbols per block. This is important because the data is reshaped into blocks of size L, and having a total number of samples that is not a multiple of L would result in incomplete blocks. By using n_valid, we can safely reshape the data into blocks without losing any information or creating incomplete blocks.
Y_real  = np.real(y[:n_valid]).reshape(NUM_BLOCKS, L) #Extract real parts and reshape into blocks of size L
Y_imag  = np.imag(y[:n_valid]).reshape(NUM_BLOCKS, L)#Extract imaginary parts and reshape into blocks of size L


X_data  = np.hstack([Y_real, Y_imag]).astype(np.float32)  

#h_block= h[:n_valid].reshape(NUM_BLOCKS, L).mean(axis=1)  # Average channel per block

train_size = int(TRAIN_RATIO * NUM_BLOCKS)   
indices = np.random.permutation(NUM_BLOCKS)

train_idx = indices[:train_size]
test_idx  = indices[train_size:]

X_train_raw = X_data[train_idx]
X_test_raw  = X_data[test_idx]

h_train = h_block[train_idx]
h_test  = h_block[test_idx]
h_true = h_test
# Compute stats from train only (no data leakage)
mu  = X_train_raw.mean(axis=0, keepdims=True)  
std = X_train_raw.std(axis=0,  keepdims=True) + 1e-8
 
X_train_norm = (X_train_raw - mu) / std
X_test_norm  = (X_test_raw  - mu) / std 
X_train = torch.tensor(X_train_norm)
X_test  = torch.tensor(X_test_norm)  
#X_train = torch.tensor(X_data[:train_size])   
#X_test  = torch.tensor(X_data[train_size:])   


train_ds     = TensorDataset(X_train, X_train) #dataset creation for train. and test.
test_ds      = TensorDataset(X_test,  X_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches/epoch : {len(train_loader)}")
print(f"Val   batches/epoch : {len(test_loader)}")

class ChannelAutoencoder(nn.Module):
  ##self.input_dim = 2*L, self.hidden = HIDDEN, self.latent = BOTTLENECK
    def __init__(self, input_dim, hidden, latent): #'self' refers to the current model object.
                                             #__init__ runs automatically when the model object is created. #ln99
        super().__init__() #initializes the parent class (nn.Module).
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), #encoder
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),#bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), #decoder
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x): #defines how input passes through the model.
        z     = self.encoder(x)    #compute compressed representation
        x_hat = self.decoder(z)    #reconstructs input from compressed representation
        return z, x_hat

model     = ChannelAutoencoder(2*L, HIDDEN, BOTTLENECK)#.to(DEVICE) #CAE is model class
                                        #creating model automatically calls __init__() which initializes the encoder and decoder layers
criterion = nn.MSELoss()     #loss function for reconstruction      
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-5)

print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}\n")

train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train() #set the model to training mode
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb, yb #.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        _, x_hat  = model(xb) #feeds a batch through the model and returns the latent representation and the reconstructed output
        loss      = criterion(x_hat, yb)  #computes reconstruction error 
        loss.backward()#backpropagation
        optimizer.step()
        running += loss.item() * xb.size(0)

    train_losses.append(running / len(train_loader.dataset))

    # Validate
    model.eval()#switch model to evaluation mode
    val_running = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb, yb #.to(DEVICE), yb.to(DEVICE)
            _, x_hat   = model(xb)
            val_running += criterion(x_hat, yb).item() * xb.size(0)

    val_losses.append(val_running / len(test_loader.dataset))
    current_val_loss = val_losses[-1]

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_losses[-1]:.6f} | "
              f"Val Loss:   {val_losses[-1]:.6f}")
 
#Channel estimation from Bottle Neck
model.eval()
with torch.no_grad():
    latent, _ = model(X_test)#.to(DEVICE)) feeds the test data into the trained autoencoder
    latent    = latent.cpu().numpy() #retrieves the latent representation from the encoder part of the autoencoder    

h_estimated = latent[:, 0] + 1j * latent[:, 1] #1st latent = ral part, 2nd latent = imag part of channel
#h_true = h_block[train_size:]  

print("\n===== True Channel Statistics =====")
print("First 10 |h_true|:", np.abs(h_true[:10]))
print("Max |h_true|:", np.max(np.abs(h_true)))
print("Min |h_true|:", np.min(np.abs(h_true)))

mse = np.mean(np.abs(h_estimated - h_true) ** 2)


print(f"\nPilot-Free Channel Estimation MSE = {mse:.6f}")




overall_nmse = mse / np.mean(np.abs(h_true)**2)
overall_nmse_db = 10 * np.log10(overall_nmse)

print(f"Overall NMSE = {overall_nmse:.6f}")
print(f"Overall NMSE(dB) = {overall_nmse_db:.2f} dB")


# NMSE vs SNR Evaluation


SNR_test = np.arange(0, 21, 2)   # 0,2,4,...20 dB

NMSE = []


model.eval()

# Use only testing channels
x_blocks = x.reshape(NUM_BLOCKS, L)

x_test_blocks = x_blocks[test_idx]
h_test_blocks = h_block[test_idx]


with torch.no_grad():

    for snr in SNR_test:

        # Noise generation for this SNR
        noise_std = 10 ** (-snr / 20)

        noise = noise_std * (
            np.random.randn(len(test_idx), L)
            + 1j*np.random.randn(len(test_idx), L)
        ) / np.sqrt(2)


        # Generate received signal
        y_test_snr = h_test_blocks[:,None] * x_test_blocks + noise


        # Convert received signal into AE input format
        Y_real = np.real(y_test_snr)
        Y_imag = np.imag(y_test_snr)

        X_test_snr = np.hstack([Y_real, Y_imag]).astype(np.float32)


        # Normalize using training statistics
        X_test_snr = (X_test_snr - mu) / std


        X_test_tensor = torch.tensor(X_test_snr)


        # Extract latent representation
        latent, _ = model(X_test_tensor)

        latent = latent.numpy()


        # Channel estimation from bottleneck
        h_est_snr = latent[:,0] + 1j*latent[:,1]


        # NMSE calculation
        nmse_snr = (
            np.mean(np.abs(h_est_snr - h_test_blocks)**2)
            /
            np.mean(np.abs(h_test_blocks)**2)
        )


        NMSE.append(nmse_snr)



NMSE = np.array(NMSE)

NMSE_dB = 10*np.log10(NMSE)
np.save("nmse_pilot_free_ae.npy", NMSE)

print("\n===== NMSE vs SNR =====")

for snr, nmse, nmse_db in zip(SNR_test, NMSE, NMSE_dB):
    print( f"SNR = {snr} dB | NMSE = {nmse:.6f} | NMSE(dB)= {nmse_db:.2f}")

#smoothing function for loss curves
def moving_avg(arr, w=5):
    return np.convolve(arr, np.ones(w) / w, mode="valid")

SMOOTH_W = 5  #smooth window size

fig, axes = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle("Autoencoder-Based Pilot-Free Channel Estimation", fontsize=13, fontweight="bold")

# Training vs Validation Loss

ax = axes[0]

train_smooth = moving_avg(train_losses, SMOOTH_W)
val_smooth   = moving_avg(val_losses, SMOOTH_W)

ax.plot(train_smooth, "b", linewidth=2.5, label="Train Loss")
ax.plot(val_smooth, "r", linewidth=2.5, label="Validation Loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Reconstruction MSE")
ax.set_title("Smoothed Training vs Validation Loss")

ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

N_PLOT = 100 # Number of blocks to plot

ax = axes[1]

ax.plot(np.abs(h_true[:N_PLOT]), "b", linewidth=1.2, label="True Channel")

ax.plot(np.abs(h_estimated[:N_PLOT]), "r--", linewidth=1.2, label="Estimated Channel")

ax.set_xlabel("Block Index")
ax.set_ylabel("Channel Magnitude")
ax.set_title("True vs Estimated Channel Magnitude")
#ax.set_xticks(np.arange(0, N_PLOT + 1, 30))#ax.set_xlim(0, 50)              # Zoom into first 50 blocks
#ax.set_yticks(np.arange(0, 2.6, 0.2))
#ax.set_xticks(np.arange(0, 51, 10))
ax.legend()
ax.grid(True)
plt.show()
plt.figure(figsize=(7,5))

plt.plot(SNR_test, NMSE, marker='o', linewidth=2)

plt.grid(True, which="both")

plt.xlabel("SNR (dB)")
plt.ylabel("NMSE")

plt.title("NMSE vs SNR")

plt.show()