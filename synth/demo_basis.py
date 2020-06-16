import numpy as np
import torch
import torch.optim as optim

from network import create_core
from AnimationPlot import animation_plot

rng = np.random.RandomState(123)

preprocess = np.load('preprocess_core.npz')

print("====================\nDataset Loaded\n====================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
window = 240

net = create_core()
net.to(device)

net.load_state_dict(torch.load("model/network_core.pth", map_location=device))

net.eval()

for i in range(10):
    Xbasis0 = np.zeros((1, 256, window//2))
    Xbasis1 = np.zeros((1, 256, window//2))
    Xbasis2 = np.zeros((1, 256, window//2))

    Xbasis0[:,i*3+0] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))
    Xbasis1[:,i*3+1] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))
    Xbasis2[:,i*3+2] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))

    Xbasis0 = (torch.from_numpy(Xbasis0)).double()
    Xbasis0 = Xbasis0.to(device)
    Xbasis1 = (torch.from_numpy(Xbasis1)).double()
    Xbasis1 = Xbasis1.to(device)
    Xbasis2 = (torch.from_numpy(Xbasis2)).double()
    Xbasis2 = Xbasis2.to(device)

    # generating pooling indices
    # pytorch can't perform unpool without indices
    # so i'm generating them myself

    temp_index_tensor = torch.zeros(1)
    i0 = []
    i1 = []
    i2 = []
    for i in range(256):
        i0.append(torch.linspace(0, 237, steps=120, dtype=torch.int64))
        i1.append(torch.linspace(1, 238, steps=120, dtype=torch.int64))
        i2.append(torch.linspace(2, 239, steps=120, dtype=torch.int64))
    b_0_indices = torch.stack(i0, dim=0)
    b_0_indices = torch.unsqueeze(b_0_indices, 0).to(device)
    b_1_indices = torch.stack(i1, dim=0)
    b_1_indices = torch.unsqueeze(b_1_indices, 0).to(device)
    b_2_indices = torch.stack(i2, dim=0)
    b_2_indices = torch.unsqueeze(b_2_indices, 0).to(device)

    with torch.no_grad():
        Xbasis0 = net(Xbasis0, b_0_indices, decode=True)
        Xbasis1 = net(Xbasis1, b_1_indices, decode=True)
        Xbasis2 = net(Xbasis2, b_2_indices, decode=True)

    Xbasis0 = Xbasis0.cpu().detach().numpy()
    Xbasis1 = Xbasis1.cpu().detach().numpy()
    Xbasis2 = Xbasis2.cpu().detach().numpy()
    
    Xbasis0 = (Xbasis0 * preprocess['Xstd']) + preprocess['Xmean']
    Xbasis1 = (Xbasis1 * preprocess['Xstd']) + preprocess['Xmean']
    Xbasis2 = (Xbasis2 * preprocess['Xstd']) + preprocess['Xmean']
        
    animation_plot([Xbasis0, Xbasis1, Xbasis2], interval=15.15)
