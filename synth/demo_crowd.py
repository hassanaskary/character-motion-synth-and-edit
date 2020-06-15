import numpy as np
import torch
import torch.optim as optim

from network import create_core, create_regressor, create_footstepper
from constraints import foot_constraint, bone_constraint, traj_constraint, constraints
from AnimationPlot import animation_plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rng = np.random.RandomState(23455)

data = np.load('../data/crowddata.npz')
preprocess = np.load('preprocess_core.npz')
preprocess_footstepper = np.load('preprocess_footstepper.npz')

print("====================\nDataset Loaded\n====================")

scenes = [
    ('scene01', 500, 1100),
    ('scene02', 500, 1100),
    ('scene03', 500, 1100),
#    ('scene04', 500, 1100)
]

off_lh, off_lt, off_rh, off_rt = 0.0, -0.25, np.pi+0.0, np.pi-0.25

batchsize = 1
epochs = 10 
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

for scene, cstart, cend in scenes:
    T = np.swapaxes(data[scene+'_Y'], 1, 2)
    T = T[:,:,cstart:cend]
    T = (T - preprocess['Xmean'][:,-7:-4]) / preprocess['Xstd'][:,-7:-4]
    T = np.concatenate([T, np.zeros((T.shape[0], 4, T.shape[2]))], axis=1)

    T = (torch.from_numpy(T)).double()
    T = T.to(device)

    with torch.no_grad():
        W = footstepper(T[:,:3])

    W = W.cpu().detach().numpy()
    T = T.cpu().detach().numpy()

    W = (W * preprocess_footstepper['Wstd']) + preprocess_footstepper['Wmean']

    offsetvar = 2*np.pi*rng.uniform(size=(T.shape[0],1,1))
    stepvar = 1.0+0.05*rng.uniform(low=-1,high=1,size=(T.shape[0],1,1))
    thesvar = 0.05*rng.uniform(low=-1,high=1,size=(T.shape[0],1,1))

    T[:,3:4] = (np.sin(np.cumsum(W[:,0:1]*stepvar,axis=2)+off_lh+offsetvar)>thesvar+np.cos(W[:,1:2])).astype(np.float)*2-1
    T[:,4:5] = (np.sin(np.cumsum(W[:,0:1]*stepvar,axis=2)+off_lt+offsetvar)>thesvar+np.cos(W[:,2:3])).astype(np.float)*2-1
    T[:,5:6] = (np.sin(np.cumsum(W[:,0:1]*stepvar,axis=2)+off_rh+offsetvar)>thesvar+np.cos(W[:,3:4])).astype(np.float)*2-1
    T[:,6:7] = (np.sin(np.cumsum(W[:,0:1]*stepvar,axis=2)+off_rt+offsetvar)>thesvar+np.cos(W[:,4:5])).astype(np.float)*2-1

    mvel = np.sqrt(np.sum(T[:,:2]**2, axis=1))
    for i in range(T.shape[0]):
        T[i,:,mvel[i]<0.75] = 1

    T = (torch.from_numpy(T)).double()
    T = T.to(device)

    with torch.no_grad():
        X, X_indices = regressor(T)
        X = core(X, X_indices, decode=True)

    X = X.cpu().detach().numpy()
    T = T.cpu().detach().numpy()

    X = (X * preprocess['Xstd']) + preprocess['Xmean']
    Xtail = (T * preprocess['Xstd'][:,-7:]) + preprocess['Xmean'][:,-7:]

    X = (torch.from_numpy(X)).double()
    X = X.to(device)

    with torch.no_grad():
        X_H, X_H_indices = core(X, encode=True)

    X_H.requires_grad_()

    optimizer= optim.Adam([X_H], lr=lr, betas=(beta1, beta2))

    for e in range(epochs):
        optimizer.zero_grad()
        loss = constraints(core, X_H, X_H_indices, preprocess, labels=Xtail[:,-4:], traj=Xtail[:,:3], to_run=("foot", "bone", "traj"))
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print("epoch: ", e, "loss: ", loss.item())

    with torch.no_grad():
        X = core(X_H, unpool_indices=X_H_indices, decode=True)

    X = X.cpu().detach().numpy()
    Xtail = Xtail.cpu().detach().numpy()

    X[:,-7:] = Xtail

    animation_plot([X[0:1,:,:200], X[10:11,:,:200], X[20:21,:,:200]], interval=15.15)

    # X = np.swapaxes(X, 1, 2)
    # joints = X[:,:,:-7].reshape((X.shape[0], X.shape[1], -1, 3))
    # joints = -Quaternions(data[scene+'_rot'][:,cstart:cend])[:,:,np.newaxis] * joints
    # joints[:,:,:,0] += data[scene+'_pos'][:,cstart:cend][:,:,np.newaxis][:,:,:,0]
    # joints[:,:,:,2] += data[scene+'_pos'][:,cstart:cend][:,:,np.newaxis][:,:,:,2]
    #np.savez_compressed('./videos/crowd/'+scene+'.npz', X=joints)
