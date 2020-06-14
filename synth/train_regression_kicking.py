import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from network import create_core, create_regressor

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_hdm05.npz')
preprocess = np.load('preprocess_core.npz')

print("====================\nDataset Loaded\n====================")

data_kicking = np.hstack([np.arange(199, 246), np.arange(862, 906), np.arange(1582,1640), np.arange(2188,2233), np.arange(2796,2844)])
rng.shuffle(data_kicking)

kicking_train = data_kicking[:len(data_kicking)//2]
kicking_valid = data_kicking[len(data_kicking)//2:]

X = data['clips'][kicking_train]
X = np.swapaxes(X, 1, 2)

X = (X - preprocess['Xmean']) / preprocess['Xstd']

feet = np.array([9,10,11,12,13,14,21,22,23,24,25,26])

Y = X[:,feet]

I = np.arange(len(X))
rng.shuffle(I)
X, Y = X[I], Y[I]

X = (torch.from_numpy(X)).double() # target
Y = (torch.from_numpy(Y)).double() # input

print("input", Y.size(), "target", X.size())

print("====================\nPreprocessing Complete\n====================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchsize = 1
epochs = 100 
lr = 0.001
beta1 = 0.9
beta2 = 0.999

regressor = create_regressor(input_channels=Y.size(1), indices=True)
regressor.to(device)
decoder = create_core()
decoder.to(device)

decoder.load_state_dict(torch.load("model/network_core.pth", map_location=device))

params = list(regressor.parameters()) + list(decoder.parameters())

criterionMSE = nn.MSELoss().to(device)
optimizer = optim.Adam(params, lr=lr, betas=(beta1, beta2))

for e in range(epochs):

    print("\n#############################################")
    print(f"Staring epoch {e}/{epochs} at {time.asctime(time.localtime(time.time()))}")
    print("#############################################\n")

    epoch_start_time = time.time()

    for index in tqdm(range(X.size(0)), desc="Iterations"): 
        target = X[index]
        target = target.view(1, target.size(0), target.size(1))
        batch = Y[index]
        batch = batch.view(1, batch.size(0), batch.size(1))
        batch.to(device)
        regressed, indices = regressor(batch)
        decoded = decoder(regressed, indices, decode=True)
        optimizer.zero_grad()
        loss = criterionMSE(decoded, target)
        loss.backward()
        optimizer.step()

    print(f"\n===> Loss: {loss.item()}")

    epoch_end_time = time.time()
    print(f"\nEpoch {e} finished. Time taken {epoch_end_time - epoch_start_time} sec")

    if e % 10 == 0:
        torch.save(regressor.state_dict(), f"model/network_regression_kick.pth")
        print("====================\nModel Saved\n====================")
