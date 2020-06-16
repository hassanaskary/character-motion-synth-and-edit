import numpy as np
import torch
import torch.optim as optim

from network import create_core
from constraints import constraints
from AnimationPlot import animation_plot

rng = np.random.RandomState(413342)

X = np.load('../data/processed/data_cmu.npz')['clips']
preprocess = np.load('preprocess_core.npz')

print("====================\nDataset Loaded\n====================")

X = np.swapaxes(X, 1, 2)
X = (X - preprocess['Xmean']) / preprocess['Xstd']

print("====================\nPreprocessing Complete\n====================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.01
beta1 = 0.9
beta2 = 0.999
batchsize = 1
epochs = 50

net = create_core()
net.to(device)

net.load_state_dict(torch.load("model/network_core.pth", map_location=device))

net.eval()

for _ in range(10):

    index = rng.randint(X.shape[0])
    Xorgi = np.array(X[index:index+1])
    Xnois = Xorgi.copy()
    Xnois[:,16*3-1:17*3] = 0.0

    Xnois = (torch.from_numpy(Xnois)).double()
    Xnois = Xnois.to(device)

    with torch.no_grad():
        Xrecn = net(Xnois)

    Xnois = Xnois.cpu().detach().numpy()
    Xrecn = Xrecn.cpu().detach().numpy()

    Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

    Xrecn = (torch.from_numpy(Xrecn)).double()
    Xrecn = Xrecn.to(device)

    Xorgi = (torch.from_numpy(Xorgi)).double()
    Xorgi = Xorgi.to(device)

    with torch.no_grad():
        Xrecn_H, Xrecn_H_indices = net(Xrecn, encode=True)

    Xrecn_H.requires_grad_()

    optimizer = optim.Adam([Xrecn_H], lr=lr, betas=(beta1, beta2))

    for e in range(epochs):
        optimizer.zero_grad()
        loss = constraints(net, Xrecn_H, Xrecn_H_indices, preprocess, labels=Xorgi[:,-4:].clone(), traj=Xorgi[:,-7:-4], to_run=("foot", "bone", "traj"))
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print("epoch: ", e, "loss: ", loss.item())

    with torch.no_grad():
        Xrecn = net(Xrecn_H, unpool_indices=Xrecn_H_indices, decode=True)

    Xrecn = Xrecn.cpu().detach().numpy()
    Xorgi = Xorgi.cpu().detach().numpy()

    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

    Xrecn[:,-7:-4] = Xorgi[:,-7:-4]

    animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
