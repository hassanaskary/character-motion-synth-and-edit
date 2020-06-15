import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from network import create_core
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


# eq 14 in 7.2
def styletransfer(H, g_phi_S, phi_C, style_amount):
    s, c = style_amount, 1.0
    s, c = s / (s + c), c / (s + c)
    g_H = compute_gram(H)
    loss = s * torch.pow(torch.norm(
        (g_phi_S - g_H), 2), 2) + c * torch.pow(torch.norm((phi_C - H), 2), 2)
    return loss


def foot_constraint(Xtrsf_H, V, labels):
    feet = torch.tensor([[12, 13, 14], [15, 16, 17], [24, 25, 26],
                         [27, 28, 29]])
    contact = (labels > 0.5)

    offsets = torch.cat([
        V[:, feet[:, 0:1]],
        torch.zeros(
            (V.size(0), len(feet), 1, V.size(2))).double(), V[:, feet[:, 2:3]]
    ], dim=2)

    def cross(A, B):
        return torch.cat([
            A[:, :, 1:2] * B[:, :, 2:3] - A[:, :, 2:3] * B[:, :, 1:2],
            A[:, :, 2:3] * B[:, :, 0:1] - A[:, :, 0:1] * B[:, :, 2:3],
            A[:, :, 0:1] * B[:, :, 1:2] - A[:, :, 1:2] * B[:, :, 0:1]
        ], dim=2)

    neg_V = torch.unsqueeze(-V[:, -5], 1)
    neg_V = torch.unsqueeze(neg_V, 1)
    rotation = neg_V * cross(torch.tensor([[[0, 1, 0]]]), offsets)

    velocity_scale = 10

    cost_feet_x = velocity_scale * torch.mean(
            contact[:, :, :-1] * (
                ((V[:, feet[:, 0], 1:] - V[:, feet[:, 0], :-1]) +
                torch.unsqueeze(V[:, -7, :-1], 1) +
                rotation[:, :, 0, :-1])**2       
            )
    )

    cost_feet_z = velocity_scale * torch.mean(
            contact[:,:,:-1] * (
                ((V[:,feet[:,2],1:] - V[:,feet[:,2],:-1]) + 
                  torch.unsqueeze(V[:,-6,:-1], 1) + 
                  rotation[:,:,2,:-1])**2
            )
    )

    #cost_feet_y = torch.mean(contact * ((V[:,feet[:,1]] - torch.tensor([[0.75], [0.0], [0.75], [0.0]]))**2))

    cost_feet_y = velocity_scale * torch.mean(
            contact[:,:,:-1] * 
            ((V[:,feet[:,1],1:] - V[:,feet[:,1],:-1])**2)
    )

    cost_feet_h = 10.0 * torch.mean(
            torch.min(
                V[:,feet[:,1],1:], 
                torch.zeros(V[:,feet[:,1],1:].size(), dtype=torch.double)
            )**2
    )

    return (cost_feet_x + cost_feet_z + cost_feet_y + cost_feet_h) / 4


def bone_constraint(Xtrsf_H, V, parents=torch.tensor([-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 12, 14, 15, 16, 12, 18, 19, 20]), lengths=torch.tensor([2.40, 7.15, 7.49, 2.36, 2.37, 7.43, 7.50, 2.41, 2.04, 2.05, 1.75, 1.76, 2.90, 4.98, 3.48, 0.71, 2.73, 5.24, 3.44, 0.62])):
    J = V[:, :-7].view((V.size(0), len(parents), 3, V.size(2)))
    
    lengths = torch.unsqueeze(lengths, 1)
    lengths = torch.unsqueeze(lengths, 0)

    return torch.mean((
            torch.sqrt(torch.sum((J[:, 2:] - J[:, parents[2:]])**2, dim=2)) -
            lengths)**2)


def traj_constraint(Xtrsf_H, V, traj):
    velocity_scale = 10
    return velocity_scale * torch.mean((V[:, -7:-4] - traj)**2)


# eq 13 in 7.1
def constraints(Xtrsf_H, Xtrsf_H_indices, labels, traj):
    preprocess_Xstd_torch = (torch.from_numpy(preprocess['Xstd'])).double()
    preprocess_Xmean_torch = (torch.from_numpy(preprocess['Xmean'])).double()

    V = net(Xtrsf_H, unpool_indices=Xtrsf_H_indices, decode=True)
    V = (V * preprocess_Xstd_torch) + preprocess_Xmean_torch

    foot = foot_constraint(Xtrsf_H, V, labels)
    bone = bone_constraint(Xtrsf_H, V)
    traj = traj_constraint(Xtrsf_H, V, traj)
    loss = (foot + bone + traj) / 3
    return loss


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
    S.to(device)
    C.to(device)

    # optimizng the style function

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

    optimizer_style = optim.Adam([H], lr=lr, betas=(beta1, beta2))

    print("====================\nOptimizing Style\n====================")

    # optimizing style transfer equation
    for e in range(epochs):
        optimizer_style.zero_grad()
        loss_style = styletransfer(H, g_phi_S, phi_C, style_amount)
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
    Xtrsf.to(device)

    Xtail = (torch.from_numpy(Xtail)).double()
    Xtail.to(device)

    with torch.no_grad():
        Xtrsf_H, Xtrsf_H_indices = net(Xtrsf, encode=True)

    # we need to learn Xtrsf_H so that constraints return minimum value
    Xtrsf_H.requires_grad_()

    optimizer_constraint = optim.Adam([Xtrsf_H], lr=lr, betas=(beta1, beta2))

    print("====================\nOptimizing Bone length, Trajectory, and Foot sliding\n====================")

    # optimizing style transfer equation
    for e in range(epochs):
        optimizer_constraint.zero_grad()
        loss_constraint = constraints(Xtrsf_H, Xtrsf_H_indices, Xtail[:, -4:], Xtail[:, :3])
        loss_constraint.backward()
        optimizer_constraint.step()
        if e % 10 == 0:
            print("epoch: ", e, "loss: ", loss_constraint.item())

    with torch.no_grad():
        Xtrsf = net(Xtrsf_H, unpool_indices=Xtrsf_H_indices, decode=True)

    Xtrsf = (Xtrsf * preprocess['Xstd']) + preprocess['Xmean']

    Xtrsf[:, -7:] = Xtail

    Xstyl = np.concatenate([Xstyl, Xstyl], axis=2)

    Xtrsf = Xtrsf.cpu().detach().numpy()

    animation_plot([Xstyl, Xcntn, Xtrsf], interval=15.15)
