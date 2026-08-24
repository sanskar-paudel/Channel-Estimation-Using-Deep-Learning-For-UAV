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
HIDDEN      = 128
EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 1e-3
#DEVICE = torch.device("cpu")
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {DEVICE}")

SNR_dB = np.random.randint(0, 21, size=NUM_SAMPLES).astype(float)

x_real = 2 * np.random.randint(0, 2, NUM_SAMPLES) - 1
x_imag = 2 * np.random.randint(0, 2, NUM_SAMPLES) - 1
x      = (x_real + 1j * x_imag) / np.sqrt(2)

h = (np.random.randn(NUM_SAMPLES) + 1j * np.random.randn(NUM_SAMPLES)) / np.sqrt(2)
# One Rayleigh channel per block
h_block = (np.random.randn(NUM_BLOCKS) + 1j * np.random.randn(NUM_BLOCKS)) / np.sqrt(2)

# Repeat the same channel L times inside each block
h = np.repeat(h_block, L)
noise_std = 10 ** (-SNR_dB / 20)
noise     = (noise_std * (np.random.randn(NUM_SAMPLES) + 1j * np.random.randn(NUM_SAMPLES)) / np.sqrt(2))

y = h * x + noise

n_valid = NUM_BLOCKS * L 
#n_valid = int(NUM_SAMPLES / L) * L## What is the need of n_valid? Isn't n_valid = NUM_SAMPLES?It is used to ensure that the number of samples used for training and testing is a multiple of L, which is the number of symbols per block. This is important because the data is reshaped into blocks of size L, and having a total number of samples that is not a multiple of L would result in incomplete blocks. By using n_valid, we can safely reshape the data into blocks without losing any information or creating incomplete blocks.
Y_real  = np.real(y[:n_valid]).reshape(NUM_BLOCKS, L) #Extract real parts and reshape into blocks of size L
Y_imag  = np.imag(y[:n_valid]).reshape(NUM_BLOCKS, L)#Extract imaginary parts and reshape into blocks of size L


X_data  = np.hstack([Y_real, Y_imag]).astype(np.float32)  

#h_blocks= h[:n_valid].reshape(NUM_BLOCKS, L).mean(axis=1)  # Average channel per block

train_size = int(TRAIN_RATIO * NUM_BLOCKS)   
X_train_raw = X_data[:train_size] #normalization on training data
X_test_raw  = X_data[train_size:]
 
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
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

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
h_true      = h_block[train_size:]   

mse = np.mean(np.abs(h_estimated - h_true) ** 2)


print(f"  Pilot-Free Channel Estimation MSE = {mse:.6f}")

#smoothing function for loss curves
def moving_avg(arr, w=5):
    return np.convolve(arr, np.ones(w) / w, mode="valid")

SMOOTH_W = 5  

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
ax.set_xticks(np.arange(0, N_PLOT + 1, 30))#ax.set_xlim(0, 50)              # Zoom into first 50 blocks
#ax.set_yticks(np.arange(0, 2.6, 0.2))
#ax.set_xticks(np.arange(0, 51, 10))
ax.legend()
ax.grid(True)
plt.show()