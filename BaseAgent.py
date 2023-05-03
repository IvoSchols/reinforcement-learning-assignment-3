from abc import ABC, abstractmethod
import torch.nn as nn
import torch 
import numpy as np
import torch.optim as optim

from AgentNetwork import ReinforceModel


class Agent(ABC):
    def __init__(self, env, device, optimizer='adam', lossFunction='huber', state_space= 98, action_space=3): #98 for 7x7 grid
        self.env = env
        self.device = device
        self.learning_rate = 1e-4
        self.gamma = 0.7
        self.epsilon = 0.1     # Determines chance of exploration  of egreedy policy
        self.temperature = 0.5 # Determines greediness of boltzmann policy
        self.beta = 0.3        # Determines chance of curiosity\

        # self.policy is the policy network -> pi(a|s)
        self.net = ReinforceModel(state_space, action_space).to(self.device)

        match optimizer:
            case 'adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
            case 'rmsprop':
                self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.learning_rate)
            case 'sgd':
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)
            case other:
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        match lossFunction:
            case 'huber':
                self.lossFunction = nn.SmoothL1Loss()
            case 'mse':
                self.lossFunction = nn.MSELoss()
            case 'crossentropy':
                self.lossFunction = nn.CrossEntropyLoss()
            case other:
                self.lossFunction = nn.SmoothL1Loss()
   




    @abstractmethod
    def optimize_model(self, state, action, next_state, reward, done, rewards):
        pass

    def select_action(self, state):      
        return self.net.forward(state)
        
    def reset_model(self):
        for layer in self.net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()