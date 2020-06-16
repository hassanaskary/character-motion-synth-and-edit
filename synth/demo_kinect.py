import numpy as np
import torch
import torch.optim as optim

from network import create_core
from constraints import constraints
from AnimationPlot import animation_plot

rng = np.random.RandomState(23455)

Xkinect = np.load('../data/processed/data_edin_kinect.npz')['clips']
Xsens = np.load('../data/processed/data_edin_xsens.npz')['clips']
preprocess = np.load('preprocess_core.npz')

print("====================\nDataset Loaded\n====================")

Xkinect = np.swapaxes(Xkinect, 1, 2)
Xsens = np.swapaxes(Xsens, 1, 2)
Xkinect = (Xkinect - preprocess['Xmean']) / preprocess['Xstd']
Xsens = (Xsens - preprocess['Xmean']) / preprocess['Xstd']

print("====================\nPreprocessing Complete\n====================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.01
beta1 = 0.9
beta2 = 0.999
batchsize = 1
epochs = 100

net = create_core()
net.to(device)

net.load_state_dict(torch.load("model/network_core.pth", map_location=device))

net.eval()

index = Xkinect.shape[0]-82
#index = rng.randint(Xkinect.shape[0].eval())

Xsensclip = np.concatenate([
    Xsens[index+0:index+1,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+1:index+2,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+2:index+3,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+3:index+4,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+4:index+5,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+5:index+6,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+6:index+7,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+7:index+8,:,Xsens.shape[2]//4:-Xsens.shape[2]//4]], axis=2)

Xkinectclip = np.concatenate([
    Xkinect[index+0:index+1,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+1:index+2,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+2:index+3,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+3:index+4,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+4:index+5,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+5:index+6,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+6:index+7,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+7:index+8,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4]], axis=2)

Xorgi = np.array(Xsensclip)
Xnois = np.array(Xkinectclip)

Xnois = (torch.from_numpy(Xnois)).double()
Xnois = Xnois.to(device)

with torch.no_grad():
    Xrecn = net(Xnois)

Xrecn = Xrecn.cpu().detach().numpy()
Xnois = Xnois.cpu().detach().numpy()

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

animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
