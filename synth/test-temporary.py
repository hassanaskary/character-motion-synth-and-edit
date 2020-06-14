import numpy as np
import torch
import network

i = np.zeros((1, 3, 240))
t = torch.from_numpy(i)
n = network.create_footstepper()
n = n.double()

re = n(t.double())
print(re.size())
