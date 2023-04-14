from abc import ABC, abstractmethod
import torch.nn as nn
from Qnetwork import DQNModel 
import torch 
import numpy as np

class Agent(ABC):
    def __init__(self, env, device, optimiser='adam', lossFunction='huber'):
        self.env = env
        self.device = device
        self.learning_rate = 1e-4
        self.gamma = 0.7
        self.epsilon = 0.1     # Determines chance of exploration  of egreedy policy
        self.temperature = 0.5 # Determines greediness of boltzmann policy
        self.beta = 0.3        # Determines chance of curiosity\

        self.net = DQNModel().to(device)
        self.icm = IntrinsicCuriosityModule(env.observation_space.shape[0], env.action_space.n, env, device).to(device)
        self.policy = 'egreedy'

        # All perform around the same if you average results over multiple runs
        if(optimiser == 'adam'):
            self.optim = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, amsgrad=True)
        elif(optimiser == 'SGD'):
            self.optim = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
        elif(optimiser == 'adagrad'):
            self.optim = torch.optim.Adagrad(self.net.parameters(), lr=self.learning_rate) 
        else:
            self.optim = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, amsgrad=True)


        # We are looking for a loss function that weights steeper angles increasingly worse!
        # Because it is harder to recover from increasing angles, so we want to avoid even the slightest deviations!
        if(lossFunction == 'huber'):
            self.lossFunction = nn.HuberLoss() # BEST: Performs better than L1Loss
        elif(lossFunction == 'smoothL1'):
            self.lossFunction = nn.SmoothL1Loss()
        elif(lossFunction == 'L1'):
            self.lossFunction = nn.L1Loss()# Performs better than SmoothL1Loss
        elif(lossFunction == 'MSE'):
            self.lossFunction = nn.MSELoss()
        elif(lossFunction == 'KLDiv'):
            self.lossFunction = nn.KLDivLoss()
        else:
            self.lossFunction = nn.HuberLoss()
        



    @abstractmethod
    def optimize_model(self, state, action, next_state, reward, done):
        pass

    def select_action(self, state):      
        if self.policy == 'egreedy' or self.policy == 'cd-exploration':
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