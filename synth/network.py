import torch
import torch.nn as nn

def create_core():
    net = Core()
    return net.double()

def create_regressor(input_channels=7, indices=False):
    net = Regressor(input_channels, indices)
    return net.double()

def create_footstepper():
    net = Footstepper()
    return net.double()

class Core(nn.Module):
    def __init__(self):
        super(Core, self).__init__()

        self.encoder = nn.Sequential(
                nn.Dropout(0.25),
                nn.Conv1d(73 , 256, 25, stride=1, padding=12, bias=True),
                nn.ReLU(),
                )

        self.pool = nn.MaxPool1d(2, stride=2, padding=0, return_indices=True)
        self.unpool = nn.MaxUnpool1d(2, stride=2, padding=0)

        self.decoder = nn.Sequential(
                nn.Dropout(0.25),
                nn.Conv1d(256, 73, 25, stride=1, padding=12, bias=True),
                nn.ReLU()
                )

    def forward(self, x, unpool_indices=0, decode=False, encode=False):
        # when RECONSTRUCTING only provide input tensor. leave all other
        # arguments as is

        # when DECODING provide input tensor, indices, and set decode to True
        # leave encode to be False

        # when ENCODING provide input tensor, and set encode to True, leave 
        # the rest as is
        if decode:
            x = self.unpool(x, unpool_indices)
            x = self.decoder(x)
            return x
        elif encode:
            x = self.encoder(x)
            x, indices = self.pool(x)
            return x, indices
        else:
            x = self.encoder(x)
            x, indices = self.pool(x)
            x = self.unpool(x, indices)
            x = self.decoder(x)
            return x

class Regressor(nn.Module):
    def __init__(self, input_channels, indices):
        super(Regressor, self).__init__()

        self.indices = indices

        self.network = nn.Sequential(
                nn.Dropout(0.25),
                nn.Conv1d(input_channels, 64, 45, stride=1, padding=22, bias=True),
                nn.ReLU(),

                nn.Dropout(0.25),
                nn.Conv1d(64, 128, 25, stride=1, padding=12, bias=True),
                nn.ReLU(),

                nn.Dropout(0.25),
                nn.Conv1d(128, 256, 15, stride=1, padding=7, bias=True),
                nn.ReLU()
                )

        self.pool = nn.MaxPool1d(2, stride=2, padding=0, return_indices=self.indices)

    def forward(self, x):
        x = self.network(x)
        if self.indices:
            x, indices = self.pool(x)
            return x, indices
        else:
            x = self.pool(x)
        return x

class Footstepper(nn.Module):
    def __init__(self):
        super(Footstepper, self).__init__()

        self.network = nn.Sequential(
                nn.Dropout(0.25),
                nn.Conv1d(3, 64, 65, stride=1, padding=32, bias=True),
                nn.ReLU(),

                nn.Dropout(0.25),
                nn.Conv1d(64, 5, 45, stride=1, padding=22, bias=True)
                )

    def forward(self, x):
        x = self.network(x)
        return x
