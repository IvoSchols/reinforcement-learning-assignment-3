from abc import ABC, abstractmethod
import torch.nn as nn
import torch 
import numpy as np

class Agent(ABC):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.learning_rate = 1e-4
        self.gamma = 0.7
        self.epsilon = 0.1     # Determines chance of exploration  of egreedy policy
        self.temperature = 0.5 # Determines greediness of boltzmann policy
        self.beta = 0.3        # Determines chance of curiosity\

        self.policy = 'egreedy'



    @abstractmethod
    def optimize_model(self, state, action, next_state, reward, done, rewards):
        pass

    def select_action(self, state):      
        if self.policy == 'egreedy':
            p = np.random.rand()
            if p < self.epsilon:
                a = self.env.action_space.sample()
            else:
                a = self.net(state).max(1)[1].view(1, 1).item()
            return a
        if self.policy == 'boltzmann':
            m = nn.Softmax()
            a_probabilities = m(self.net(state).squeeze() / self.temperature)
            
            a_distribution = torch.distributions.categorical.Categorical(a_probabilities.data)
            a = a_distribution.sample()
            a = a.item()
            return a
        
    def reset_model(self):
        for layer in self.net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()