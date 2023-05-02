import torch
import torch.nn as nn

class ActroCriticModel(nn.Module):
    
    def __init__(self, num_inputs):
        self.hidden_size = 64
        self.critic_linear1 = nn.Linear(num_inputs, self.hidden_size)
        self.critic_linear2 = nn.Linear(self.hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, )
        self.actor_linear2 = nn.Linear(self.hidden_size, 3)
    
    def forward(self, state):
        value = torch.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = torch.relu(self.actor_linear1(state))
        policy_dist = torch.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist