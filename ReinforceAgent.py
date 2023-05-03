
import torch 

import numpy as np
import torch.nn as nn
from BaseAgent import Agent
from AgentNetwork import ReinforceModel

class ReinforceAgent(Agent):

    def __init__(self, env, device, optimizer='adam', lossFunction='huber', state_space= 98, action_space=3):
        # state_space = env.observation_space.shape[0]
        super().__init__(env, device, optimizer, lossFunction, state_space, action_space)
        

    def optimize_model(self, state, action, next_state, reward, done, rewards):
        # TODO: evaluate correctness of this function
        log_probs = torch.stack(log_probs)
        loss = -torch.mean(log_probs) * (sum(rewards) - 15)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return reward
    


