import torch
import torch.nn as nn
import torch.nn.functional as F


class ReinforceModel(nn.Module):
    def __init__(self,num_input):
        super(ReinforceModel,self).__init__()

        self.layer1 = nn.Linear(num_input,64) # 98 for 7x7
        self.layer2 = nn.Linear(64,3) #3 actions
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return torch.softmax(x, dim=-1)