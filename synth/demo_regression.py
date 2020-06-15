import numpy as np
import torch
import torch.optim as optim

from network import create_core, create_regressor, create_footstepper
from constraints import foot_constraint, bone_constraint, traj_constraint, constraints
from AnimationPlot import animation_plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

preprocess = np.load('preprocess_core.npz')
preprocess_footstepper = np.load('preprocess_footstepper.npz')

print("====================\nDataset Loaded\n====================")

# alpha - user parameter scaling the frequency of stepping.
#         Higher causes more stepping so that 1.25 adds a 
#         quarter more steps. 1 is the default (output of
#         footstep generator)
#
# beta - Factor controlling step duration. Increasing reduces 
#        the step duration. Small increases such as 0.1 or 0.2 can
#        cause the character to run or jog at low speeds. Small 
#        decreases such as -0.1 or -0.2 can cause the character 
#        to walk at high speeds. Too high values (such as 0.5) 
#        may cause the character to skip steps completely which 
#        can look bad. Default is 0.
#
#alpha, beta = 1.25, 0.1
alpha, beta = 1.0, 0.0

# controls minimum/maximum duration of steps
minstep, maxstep = 0.9, -0.5

off_lh, off_lt, off_rh, off_rt = 0.0, -0.1, np.pi+0.0, np.pi-0.1

indices = [(30, 15*480), (60, 15*480), (90, 15*480)]

batchsize = 1
epochs = 250 
lr = 0.001
beta1 = 0.9
beta2 = 0.999

regressor = create_regressor(7, indices=True)
regressor.to(device)

core = create_core()
core.to(device)

footstepper = create_footstepper()
footstepper.to(device)

core.load_state_dict(torch.load("model/network_core.pth", map_location=device))
regressor.load_state_dict(torch.load("model/network_regression.pth", map_location=device))
footstepper.load_state_dict(torch.load("model/network_footstepper.pth", map_location=device))

core.eval()
regressor.eval()
footstepper.eval()

for index, length in indices:
    Torig = np.load('../data/curves.npz')['C'][:,:,index:index+length]
    Torig = (Torig - preprocess['Xmean'][:,-7:-4]) / preprocess['Xstd'][:,-7:-4]
    
    Torig = (torch.from_numpy(Torig)).double()
    Torig = Torig.to(device)

    with torch.no_grad():
        W = footstepper(Torig[:,:3])

    Torig = Torig.cpu().detach().numpy()
    W = W.cpu().detach().numpy()

    W = (W * preprocess_footstepper['Wstd']) + preprocess_footstepper['Wmean']

    Torig = (np.concatenate([Torig,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lh)>np.clip(np.cos(W[:,1:2])+beta, maxstep, minstep))*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lt)>np.clip(np.cos(W[:,2:3])+beta, maxstep, minstep))*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rh)>np.clip(np.cos(W[:,3:4])+beta, maxstep, minstep))*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rt)>np.clip(np.cos(W[:,4:5])+beta, maxstep, minstep))*2-1], axis=1))

    Torig = (torch.from_numpy(Torig)).double()
    Torig = Torig.to(device)

    with torch.no_grad():
        Xrecn, Xrecn_H = regressor(Torig)
        Xrecn = core(Xrecn, Xrecn_H, decode=True)

    Xrecn = Xrecn.cpu().detach().numpy()
    Torig = Torig.cpu().detach().numpy()

    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Xtraj = ((Torig * preprocess['Xstd'][:,-7:]) + preprocess['Xmean'][:,-7:]).clone()

    Xnonc = Xrecn.copy()

    Xrecn = (torch.from_numpy(Xrecn)).double()
    Xrecn = Xrecn.to(device)

    with torch.no_grad():
        Xrecn_H, Xrecn_H_indices = core(Xrecn, encode=True)

    Xrecn_H.requires_grad_()

    optimizer= optim.Adam([Xrecn_H], lr=lr, betas=(beta1, beta2))

    for e in range(epochs):
        optimizer.zero_grad()
        loss = constraints(core, Xrecn_H, Xrecn_H_indices, preprocess, labels=Xtraj[:,-4:], traj=Xtraj[:,:3], to_run=("foot", "bone", "traj"))
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print("epoch: ", e, "loss: ", loss.item())

    with torch.no_grad():
        Xrecn = core(Xrecn_H, unpool_indices=Xrecn_H_indices, decode=True)

    Xrecn = Xrecn.cpu().detach().numpy()

    Xrecn[:,-7:] = Xtraj
    
    animation_plot([Xnonc, Xrecn], interval=15.15)
