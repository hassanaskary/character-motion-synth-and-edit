import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from network import create_core
from utils import compute_gram

rng = np.random.RandomState(23455)

#Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']

print("====================\nDataset Loaded\n====================")

#Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2)
Xhdm05 = np.swapaxes(Xhdm05, 1, 2)
Xedin_locomotion = np.swapaxes(Xedin_locomotion, 1, 2)
Xedin_misc = np.swapaxes(Xedin_misc, 1, 2)

preprocess = np.load('preprocess_core.npz')

#Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']
Xhdm05 = (Xhdm05 - preprocess['Xmean']) / preprocess['Xstd']
Xedin_locomotion = (Xedin_locomotion - preprocess['Xmean']) / preprocess['Xstd']
Xedin_misc = (Xedin_misc - preprocess['Xmean']) / preprocess['Xstd']

# content_clip, content_database, style_clip, style_database, style_amount
pairings = [
   (245, Xedin_locomotion, 100, Xedin_misc,  0.1),
   (242, Xedin_locomotion, 110, Xedin_misc,  0.1),
   (238, Xedin_locomotion,  90, Xedin_misc,  0.1),
   
   ( 53, Xedin_locomotion,  50, Xedin_misc,  0.1),
   ( 52, Xedin_locomotion,  71, Xedin_misc,  0.1),
   ( 15, Xedin_locomotion,  30, Xedin_misc,  0.1),
    
#   (234, Xedin_locomotion,  39, Xstyletransfer,  0.1),
#   (230, Xedin_locomotion, 321, Xstyletransfer,  0.1),
#   (225, Xedin_locomotion,  -41, Xstyletransfer, 0.1),
   
#   (220, Xedin_locomotion,  148, Xstyletransfer,  0.1),
]

print("====================\nPreprocessing Complete\n====================")

# eq 14 in 7.2
def styletransfer(H, g_phi_S, phi_C, style_amount):
    s, c =  style_amount, 1.0
    s, c = s / (s + c), c / (s + c)
    g_H = compute_gram(H)
    Xtrsf = s * torch.pow(torch.norm((g_phi_S - g_H), 2), 2) + c * torch.pow(torch.norm((phi_C - H), 2), 2)
    return Xtrsf
    

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

for content_clip, content_database, style_clip, style_database, style_amount in pairings:
    
    # S = style and C = content
    S = style_database[style_clip:style_clip+1]
    C = np.concatenate([
        content_database[content_clip+0:content_clip+1,:,S.shape[2]//4:-S.shape[2]//4],
        content_database[content_clip+1:content_clip+2,:,S.shape[2]//4:-S.shape[2]//4],
        content_database[content_clip+2:content_clip+3,:,S.shape[2]//4:-S.shape[2]//4],
        content_database[content_clip+3:content_clip+4,:,S.shape[2]//4:-S.shape[2]//4]], axis=2)

    # the original unnormalzied style and content 
    Xstyl = (S * preprocess['Xstd']) + preprocess['Xmean']
    Xcntn = (C * preprocess['Xstd']) + preprocess['Xmean']

    S = (torch.from_numpy(S)).double()
    C = (torch.from_numpy(C)).double()
    S.to(device)
    C.to(device)

    # N is white noise
    N = np.random.normal(size=C.shape)
    N = (torch.from_numpy(N)).double()
    N.to(device)

    # encoding S and C
    with torch.no_grad():
        phi_S, phi_S_indices = net(S, encode=True)
        phi_C, phi_C_indices = net(C, encode=True)
        H, H_indices = net(N, encode=True)

    # computing gram matrix of encoded S
    g_phi_S = compute_gram(phi_S)

    # H is the hidden units that need to be learned to minimize styletransfer
    H.requires_grad_()

    optimizer = optim.Adam([H], lr=lr, betas=(beta1, beta2))

    # optimizing style transfer equation
    for e in range(epochs):
        optimizer.zero_grad()
        loss = styletransfer(H, g_phi_S, phi_C, style_amount)
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print("epoch: ", e, "loss: ", round(loss.item()))
    
    # decoding the learned hidden unit H
    with torch.no_grad():
        Xtrsf = net(H, unpool_indices=H_indices, decode=True)

    Xtrsf = (Xtrsf * preprocess['Xstd']) + preprocess['Xmean']
