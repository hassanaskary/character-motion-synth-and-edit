import numpy as np
import torch

from network import create_core
from AnimationPlot import animation_plot

rng = np.random.RandomState(23455)

X = np.load('../data/processed/data_edin_locomotion.npz')['clips']
preprocess = np.load('preprocess_core.npz')

print("====================\nDataset Loaded\n====================")

X = np.swapaxes(X, 1, 2)
X = (X - preprocess['Xmean']) / preprocess['Xstd']

print("====================\nPreprocessing Complete\n====================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = create_core()
net.to(device)

net.load_state_dict(torch.load("model/network_core.pth", map_location=device))

net.eval()

for i in range(5):

    index = rng.randint(X.shape[0]-1)
    Xorgi = np.array(X[index:index+1])

    X = (torch.from_numpy(X)).double()
    X = X.to(device)

    with torch.no_grad():
        Xrecn = net(X[index:index+1])

    Xrecn = Xrecn.cpu().detach().numpy()

    Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    
    animation_plot([Xorgi, Xrecn], interval=15.15)

    X = X.cpu().detach().numpy()
