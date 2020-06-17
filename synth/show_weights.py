import numpy as np
import matplotlib.pyplot as plt
import torch

from network import create_core

net = create_core()
net.load_state_dict(torch.load("model/network_core.pth"))
net.eval()

for li, layer in enumerate(list(net.encoder) + list(net.decoder)):
    if not isinstance(layer, torch.nn.Conv1d): continue
    
    print(li, layer.weight.size())
    shape = layer.weight.size()
    num = min(shape[0], 64)
    dims = 4, num // 4

    if shape[1] < shape[2]:
        dims = dims[1], dims[0]

    fig, axarr = plt.subplots(dims[0], dims[1], sharex=False, sharey=False)

    W = (layer.weight).cpu().detach().numpy()

    for i in range(dims[0]): 
        for j in range(dims[1]):

            axarr[i][j].imshow(
                W[i*dims[1]+j], 
                interpolation='nearest', cmap='rainbow',
                vmin=W.mean() - 5*W.std(), vmax=W.mean() + 5*W.std())

            axarr[i][j].autoscale(False)
            axarr[i][j].grid(False)
            plt.setp(axarr[i][j].get_xticklabels(), visible=False)
            plt.setp(axarr[i][j].get_yticklabels(), visible=False)
    
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.suptitle('Layer %i Filters' % li, size=16)
    plt.show()
