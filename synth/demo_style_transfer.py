import numpy as np
import torch
import torch.optim as optim

from network import create_core
from constraints import style_transfer, foot_constraint, bone_constraint, traj_constraint, constraints
from utils import compute_gram
from AnimationPlot import animation_plot

rng = np.random.RandomState(23455)

#Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
Xedin_locomotion = np.load(
    '../data/processed/data_edin_locomotion.npz')['clips']
Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']

print("====================\nDataset Loaded\n====================")

#Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2)
Xhdm05 = np.swapaxes(Xhdm05, 1, 2)
Xedin_locomotion = np.swapaxes(Xedin_locomotion, 1, 2)
Xedin_misc = np.swapaxes(Xedin_misc, 1, 2)

preprocess = np.load('preprocess_core.npz')

#Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']
Xhdm05 = (Xhdm05 - preprocess['Xmean']) / preprocess['Xstd']
Xedin_locomotion = (Xedin_locomotion -
                    preprocess['Xmean']) / preprocess['Xstd']
Xedin_misc = (Xedin_misc - preprocess['Xmean']) / preprocess['Xstd']

# content_clip, content_database, style_clip, style_database, style_amount
pairings = [
    (245, Xedin_locomotion, 100, Xedin_misc, 0.1),
    (242, Xedin_locomotion, 110, Xedin_misc, 0.1),
    (238, Xedin_locomotion, 90, Xedin_misc, 0.1),
    (53, Xedin_locomotion, 50, Xedin_misc, 0.1),
    (52, Xedin_locomotion, 71, Xedin_misc, 0.1),
    (15, Xedin_locomotion, 30, Xedin_misc, 0.1),

    #   (234, Xedin_locomotion,  39, Xstyletransfer,  0.1),
    #   (230, Xedin_locomotion, 321, Xstyletransfer,  0.1),
    #   (225, Xedin_locomotion,  -41, Xstyletransfer, 0.1),

    #   (220, Xedin_locomotion,  148, Xstyletransfer,  0.1),
]

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

for content_clip, content_database, style_clip, style_database, style_amount in pairings:

    # S = style and C = content
    S = style_database[style_clip:style_clip + 1]
    C = np.concatenate([
        content_database[content_clip + 0:content_clip + 1, :, S.shape[2] //
                         4:-S.shape[2] // 4],
        content_database[content_clip + 1:content_clip + 2, :, S.shape[2] //
                         4:-S.shape[2] // 4],
        content_database[content_clip + 2:content_clip + 3, :, S.shape[2] //
                         4:-S.shape[2] // 4],
        content_database[content_clip + 3:content_clip + 4, :, S.shape[2] //
                         4:-S.shape[2] // 4]
    ],
                       axis=2)

    # the original unnormalzied style and content
    Xstyl = (S * preprocess['Xstd']) + preprocess['Xmean']
    Xcntn = (C * preprocess['Xstd']) + preprocess['Xmean']

    S = (torch.from_numpy(S)).double()
    C = (torch.from_numpy(C)).double()
    S = S.to(device)
    C = C.to(device)

    # optimizng the style function

    # N is white noise
    N = np.random.normal(size=C.shape)
    N = (torch.from_numpy(N)).double()
    N = N.to(device)

    # encoding S and C
    with torch.no_grad():
        phi_S, phi_S_indices = net(S, encode=True)
        phi_C, phi_C_indices = net(C, encode=True)
        H, H_indices = net(N, encode=True)

    # computing gram matrix of encoded S
    g_phi_S = compute_gram(phi_S)

    # H is the hidden units that need to be learned to minimize styletransfer
    H.requires_grad_()

    optimizer_style = optim.Adam([H], lr=lr, betas=(beta1, beta2))

    print("====================\nOptimizing Style\n====================")

    # optimizing style transfer equation
    for e in range(epochs):
        optimizer_style.zero_grad()
        loss_style = style_transfer(H, g_phi_S, phi_C, style_amount)
        loss_style.backward()
        optimizer_style.step()
        if e % 10 == 0:
            print("epoch: ", e, "loss: ", loss_style.item())

    # decoding the learned hidden unit H
    with torch.no_grad():
        Xtrsf = net(H, unpool_indices=H_indices, decode=True)

    Xtrsf = Xtrsf.cpu().detach().numpy()
    Xtrsf = (Xtrsf * preprocess['Xstd']) + preprocess['Xmean']

    # optimizing the foot, bone, trajectory contriants

    Xtrsfvel = np.mean(np.sqrt(Xtrsf[:, -7:-6]**2 + Xtrsf[:, -6:-5]**2),
                       axis=2)[:, :, np.newaxis]
    Xcntnvel = np.mean(np.sqrt(Xcntn[:, -7:-6]**2 + Xcntn[:, -6:-5]**2),
                       axis=2)[:, :, np.newaxis]

    Xtail = Xtrsfvel * (Xcntn[:, -7:] / Xcntnvel)
    Xtail[:, -5:] = Xcntn[:, -5:]

    Xtrsf = (Xtrsf - preprocess['Xmean']) / preprocess['Xstd']

    Xtrsf = (torch.from_numpy(Xtrsf)).double()
    Xtrsf = Xtrsf.to(device)

    Xtail = (torch.from_numpy(Xtail)).double()
    Xtail = Xtail.to(device)

    with torch.no_grad():
        Xtrsf_H, Xtrsf_H_indices = net(Xtrsf, encode=True)

    # we need to learn Xtrsf_H so that constraints return minimum value
    Xtrsf_H.requires_grad_()

    optimizer_constraint = optim.Adam([Xtrsf_H], lr=lr, betas=(beta1, beta2))

    print("====================\nOptimizing Bone length, Trajectory, and Foot sliding\n====================")

    # optimizing style transfer equation
    for e in range(epochs):
        optimizer_constraint.zero_grad()
        loss_constraint = constraints(net, Xtrsf_H, Xtrsf_H_indices, preprocess, labels=Xtail[:, -4:], traj=Xtail[:, :3], to_run=("foot", "bone", "traj"))
        loss_constraint.backward()
        optimizer_constraint.step()
        if e % 10 == 0:
            print("epoch: ", e, "loss: ", loss_constraint.item())

    with torch.no_grad():
        Xtrsf = net(Xtrsf_H, unpool_indices=Xtrsf_H_indices, decode=True)

    Xtrsf = Xtrsf.cpu().detach().numpy()
    Xtail = Xtail.cpu().detach().numpy()

    Xtrsf = (Xtrsf * preprocess['Xstd']) + preprocess['Xmean']

    Xtrsf[:, -7:] = Xtail

    Xstyl = np.concatenate([Xstyl, Xstyl], axis=2)

    animation_plot([Xstyl, Xcntn, Xtrsf], interval=15.15)
