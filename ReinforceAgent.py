
import torch 

import numpy as np
import torch.nn as nn
from BaseAgent import Agent
from ReinforceNetwork import ReinforceModel
import torch.optim as optim

class ReinforceAgent(Agent):

    def __init__(self, env, device, optimiser='adam', lossFunction='huber'):
        super().__init__(env, device, optimiser, lossFunction)
        self.loss_fn = nn.CrossEntropyLoss()
        self.policy = ReinforceModel()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

    def optimize_model(self, state, action, next_state, reward, done, rewards):
        log_probs = torch.stack(log_probs)
        loss = -torch.mean(log_probs) * (sum(rewards) - 15)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return reward
    


