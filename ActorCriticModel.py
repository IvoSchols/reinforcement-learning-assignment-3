import torch
import torch.nn as nn

class ActorCriticModel(nn.Module):
    
    def __init__(self, num_inputs, num_actions = 3, hidden_size = 64):
        super(ActorCriticModel, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1)
        )
    
    def forward(self, state):
        value = self.critic(state)
        action_probs = self.actor(state)
        return action_probs, value