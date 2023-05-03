import torch
import torch.nn as nn



class ReinforceModel(nn.Module):
    def __init__(self,state_space, action_space):
        super(ReinforceModel,self).__init__()

        self.layer1 = nn.Linear(state_space,64) # 98 for 7x7
        self.layer2 = nn.Linear(64,action_space) #3 actions
        
    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = self.layer2(x)
        actions = torch.softmax(x)
        action = torch.multinomial(actions,1)
        log_prob = torch.log(actions[action])
        return action, log_prob