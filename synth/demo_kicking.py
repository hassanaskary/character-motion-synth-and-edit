import numpy as np
import torch
import torch.optim as optim

from network import create_core, create_regressor
from constraints import foot_constraint, bone_constraint, traj_constraint, constraints
from AnimationPlot import animation_plot

rng = np.random.RandomState(23455)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = np.load('../data/processed/data_hdm05.npz')
preprocess = np.load('preprocess_core.npz')

print("====================\nDataset Loaded\n====================")

data_kicking = np.hstack([np.arange(199, 246), np.arange(862, 906), np.arange(1582,1640), np.arange(2188,2233), np.arange(2796,2844)])
rng.shuffle(data_kicking)

kicking_train = data_kicking[:len(data_kicking)//2]
kicking_valid = data_kicking[len(data_kicking)//2:]

X = data['clips'][kicking_valid]
X = np.swapaxes(X, 1, 2)

X = (X - preprocess['Xmean']) / preprocess['Xstd']

feet = np.array([9,10,11,12,13,14,21,22,23,24,25,26])

Y = X[:,feet]

Y = (torch.from_numpy(Y)).double()
Y = Y.to(device)

print("====================\nPreprocessing Complete\n====================")

batchsize = 1
epochs = 50 
lr = 0.001
beta1 = 0.9
beta2 = 0.999

regressor = create_regressor(input_channels=Y.size(1), indices=True)
regressor.to(device)
net = create_core()
net.to(device)

net.load_state_dict(torch.load("model/network_core.pth", map_location=device))
regressor.load_state_dict(torch.load("model/network_regression_kick.pth", map_location=device))

net.eval()
regressor.eval()

for i in range(len(X)):
    with torch.no_grad():
        Xrecn, Xrecn_indices = regressor(Y[i:i+1])
        Xrecn = net(Xrecn, Xrecn_indices, decode=True)

    Xorig = np.array(X[i:i+1])
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    
    Xrecn = Xrecn.cpu().detach().numpy()
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

    Xrecn = (torch.from_numpy(Xrecn)).double()
    Xrecn = Xrecn.to(device)

    with torch.no_grad():
        Xrecn_H, Xrecn_H_indices = net(Xrecn, encode=True)

    Xrecn_H.requires_grad_()

    optimizer= optim.Adam([Xrecn_H], lr=lr, betas=(beta1, beta2))

    for e in range(epochs):
        optimizer.zero_grad()
        loss = constraints(net, Xrecn_H, Xrecn_H_indices, preprocess, labels=Xrecn[:,-4:].clone(), to_run=("foot", "bone"))
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print("epoch: ", e, "loss: ", loss.item())

    with torch.no_grad():
        Xrecn = net(Xrecn_H, unpool_indices=Xrecn_H_indices, decode=True)

    Xrecn = Xrecn.cpu().detach().numpy()
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

    animation_plot([Xorig, Xrecn], interval=15.15)
