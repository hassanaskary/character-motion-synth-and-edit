import torch
from utils import compute_gram

# eq 14 in 7.2
def style_transfer(H, g_phi_S, phi_C, style_amount):
    s, c = style_amount, 1.0
    s, c = s / (s + c), c / (s + c)
    g_H = compute_gram(H)
    loss = s * torch.pow(torch.norm(
        (g_phi_S - g_H), 2), 2) + c * torch.pow(torch.norm((phi_C - H), 2), 2)
    return loss


def foot_constraint(V, labels):
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


def bone_constraint(V, parents=torch.tensor([-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 12, 14, 15, 16, 12, 18, 19, 20]), lengths=torch.tensor([2.40, 7.15, 7.49, 2.36, 2.37, 7.43, 7.50, 2.41, 2.04, 2.05, 1.75, 1.76, 2.90, 4.98, 3.48, 0.71, 2.73, 5.24, 3.44, 0.62])):
    J = V[:, :-7].view((V.size(0), len(parents), 3, V.size(2)))
    
    lengths = torch.unsqueeze(lengths, 1)
    lengths = torch.unsqueeze(lengths, 0)

    return torch.mean((
            torch.sqrt(torch.sum((J[:, 2:] - J[:, parents[2:]])**2, dim=2)) -
            lengths)**2)


def traj_constraint(V, traj):
    velocity_scale = 10
    return velocity_scale * torch.mean((V[:, -7:-4] - traj)**2)


# eq 13 in 7.1
def constraints(net, X, X_indices, preprocess, labels=0, traj=0, to_run=("foot", "bone", "traj")):
    preprocess_Xstd_torch = (torch.from_numpy(preprocess['Xstd'])).double()
    preprocess_Xmean_torch = (torch.from_numpy(preprocess['Xmean'])).double()

    V = net(X, unpool_indices=X_indices, decode=True)
    V = (V * preprocess_Xstd_torch) + preprocess_Xmean_torch

    foot = 0
    bone = 0
    traj = 0
    
    for choice in to_run:
        if choice == "foot":
            foot += foot_constraint(V, labels)
        elif choice == "bone":
            bone += bone_constraint(V)
        elif choice == "traj":
            traj += traj_constraint(V, traj)

    loss = (foot + bone + traj) / len(to_run)
    return loss
