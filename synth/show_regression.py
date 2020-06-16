import numpy as np
import torch

from network import create_core, create_regressor
from AnimationPlot import animation_plot

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_edin_locomotion.npz')['clips']
preprocess = np.load('preprocess_core.npz')

print("====================\nDataset Loaded\n====================")

I = np.arange(len(data))
rng.shuffle(I)

data_train = data[I[:len(data)//2]]
data_valid = data[I[len(data)//2:]]

X = data_valid
X = np.swapaxes(X, 1, 2)

X = (X - preprocess['Xmean']) / preprocess['Xstd']
Y = X[:,-7:]

print("====================\nPreprocessing Complete\n====================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

regressor = create_regressor(7, indices=True)
regressor.to(device)

decoder = create_core()
decoder.to(device)

decoder.load_state_dict(torch.load("model/network_core.pth", map_location=device))
regressor.load_state_dict(torch.load("model/network_regression.pth", map_location=device))

decoder.eval()
regressor.eval()

for i in range(5):
    index = rng.randint(len(X)-1)
    Xorig = np.array(X[index:index+1])
    
    Y = (torch.from_numpy(Y)).double()
    Y = Y.to(device)

    with torch.no_grad():
        Xrecn, Xrecn_indices = regressor(Y[index:index+1])
        Xrecn = decoder(Xrecn, Xrecn_indices, decode=True)

    Xrecn = Xrecn.cpu().detach().numpy()

    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

    Xrecn[:,-7:] = Xorig[:,-7:]
    
    animation_plot([Xorig, Xrecn], interval=15.15)

    Y = Y.cpu().detach().numpy()
