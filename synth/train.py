import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from network import create_core

rng = np.random.RandomState(23456)

Xcmu = np.load('../data/processed/data_cmu.npz')['clips']
Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
Xmhad = np.load('../data/processed/data_mhad.npz')['clips']
#Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
Xedin_xsens = np.load('../data/processed/data_edin_xsens.npz')['clips']
Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']
Xedin_punching = np.load('../data/processed/data_edin_punching.npz')['clips']

print("====================\nDataset Loaded\n====================")

#X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
X = np.swapaxes(X, 1, 2)

feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])

Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xmean[:,-7:-4] = 0.0
Xmean[:,-4:]   = 0.5

Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
Xstd[:,feet]  = 0.9 * Xstd[:,feet]
Xstd[:,-7:-5] = 0.9 * X[:,-7:-5].std()
Xstd[:,-5:-4] = 0.9 * X[:,-5:-4].std()
Xstd[:,-4:]   = 0.5

X = (X - Xmean) / Xstd

I = np.arange(len(X))
rng.shuffle(I); X = X[I]

X = (torch.from_numpy(X)).double()

print("====================\nPreprocessing Complete\n====================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
epochs = 100
lr = 0.001
beta1 = 0.9
beta2 = 0.999

net = create_core()
net.to(device)
net.train()

criterionMSE = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))

for e in range(epochs):

    print("\n#############################################")
    print(f"Staring epoch {e}/{epochs} at {time.asctime(time.localtime(time.time()))}")
    print("#############################################\n")

    epoch_start_time = time.time()

    for batch in tqdm(X, desc="Iterations"): 
        batch = batch.view(1, batch.size(0), batch.size(1))
        batch = batch.to(device)
        out = net(batch)
        optimizer.zero_grad()
        loss = criterionMSE(out, batch)
        loss.backward()
        optimizer.step()

    print(f"\n===> Loss: {loss.item()}")

    epoch_end_time = time.time()
    print(f"\nEpoch {e} finished. Time taken {epoch_end_time - epoch_start_time} sec")

    if e % 10 == 0:
        torch.save(net.state_dict(), f"model/network_core.pth")
        print("====================\nModel Saved\n====================")
