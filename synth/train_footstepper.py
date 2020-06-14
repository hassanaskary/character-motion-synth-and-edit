import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from network import create_footstepper

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_edin_locomotion.npz')['clips']
preprocess = np.load('preprocess_core.npz')

print("====================\nDataset Loaded\n====================")

I = np.arange(len(data))
rng.shuffle(I)

data_train = data[I[:len(data)//2]]
data_valid = data[I[len(data)//2:]]

X = data_train
X = np.swapaxes(X, 1, 2)

X = (X - preprocess['Xmean']) / preprocess['Xstd']

T = X[:,-7:-4]
F = X[:,-4:]

W = np.zeros((F.shape[0], 5, F.shape[2]))

for i in range(len(F)):
    
    w = np.zeros(F[i].shape)
    
    for j in range(F[i].shape[0]):
        last = -1
        for k in range(1, F[i].shape[1]):
            if last == -1 and F[i,j,k-1] < 0 and F[i,j,k-0] > 0: last = k; continue
            if last == -1 and F[i,j,k-1] > 0 and F[i,j,k-0] < 0: last = k; continue
            if F[i,j,k-1] > 0 and F[i,j,k-0] < 0:
                if k-last+1 > 10 and k-last+1 < 60:
                    w[j,last:k+1] = np.pi/(k-last)
                else:
                    w[j,last:k+1] = w[j,last-1]
                last = k
                continue
            if F[i,j,k-1] < 0 and F[i,j,k-0] > 0:
                if k-last+1 > 10 and k-last+1 < 60:
                    w[j,last:k+1] = np.pi/(k-last)
                else:
                    w[j,last:k+1] = w[j,last-1]
                last = k
                continue
    
    c = np.zeros(F[i].shape)
    
    for k in range(0, F[i].shape[1]):
        window = slice(max(k-100,0),min(k+100,F[i].shape[1]))
        ratios = (
            np.mean((F[i,:,window]>0).astype(np.float), axis=1) / 
            np.mean((F[i,:,window]<0).astype(np.float), axis=1))
        ratios[ratios==np.inf] = 100
        c[:,k] = ((np.pi*ratios) / (1+ratios))
    
    w[w==0.0] = np.nan_to_num(w[w!=0.0].mean())
    
    W[i,0:1] = w.mean(axis=0)
    W[i,1:5] = c
    
Wmean = W.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Wstd = W.std(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
W = (W - Wmean) / Wstd

np.savez_compressed('preprocess_footstepper.npz', Wmean=Wmean, Wstd=Wstd)

I = np.arange(len(T))
rng.shuffle(I)
T, F, W = T[I], F[I], W[I]

T = (torch.from_numpy(T)).double() # input
W = (torch.from_numpy(W)).double() # target

print("input", T.size(), "target", W.size())

print("====================\nPreprocessing Complete\n====================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchsize = 1
epochs = 100 
lr = 0.001
beta1 = 0.9
beta2 = 0.999

net = create_footstepper()
net.to(device)
net.train()

criterionMSE = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))

for e in range(epochs):

    print("\n#############################################")
    print(f"Staring epoch {e}/{epochs} at {time.asctime(time.localtime(time.time()))}")
    print("#############################################\n")

    epoch_start_time = time.time()

    for index in tqdm(range(W.size(0)), desc="Iterations"): 
        target = W[index]
        target = target.view(1, target.size(0), target.size(1))
        batch = T[index]
        batch = batch.view(1, batch.size(0), batch.size(1))
        batch.to(device)
        out = net(batch)
        optimizer.zero_grad()
        loss = criterionMSE(out, target)
        loss.backward()
        optimizer.step()

    print(f"\n===> Loss: {loss.item()}")

    epoch_end_time = time.time()
    print(f"\nEpoch {e} finished. Time taken {epoch_end_time - epoch_start_time} sec")

    if e % 10 == 0:
        torch.save(net.state_dict(), f"model/network_footstepper.pth")
        print("====================\nModel Saved\n====================")
