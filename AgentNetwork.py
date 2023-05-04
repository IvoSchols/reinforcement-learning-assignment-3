import torch
import torch.nn as nn
import torch.nn.functional as F



class ReinforceModel(nn.Module):
    def __init__(self,state_space, action_space):
        super(ReinforceModel,self).__init__()

        # State space is 98 for 7x7
        # Action spaec is 3
        self.layers = nn.Sequential(
            nn.Linear(state_space,128), 
            nn.ReLU(), 
            nn.Linear(128,action_space),
            nn.Softmax(dim=0))

        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x