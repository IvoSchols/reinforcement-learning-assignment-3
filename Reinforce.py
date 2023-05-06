import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from catch import Catch
import ray


class ReinforceModel(nn.Module):
    def __init__(self,state_space=98, action_space=3, hidden_size=64):
        super(ReinforceModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_space,hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size,action_space),
            nn.Softmax(dim=0))
        
    def forward(self, x):
        return self.layers(x)

class ReinforceAgent:
    def __init__(self, env, device, hidden_size=64, learning_rate=0.01, gamma=0.99, entropy_weight=0.01, n_steps=5):
        self.env = Catch()
        self.device = device
        self.num_inputs = np.prod(env.observation_space.shape)
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.n_steps = n_steps

        self.model = ReinforceModel(self.num_inputs, self.num_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)


    def select_action(self, state):
        state = torch.tensor(state).to(device)
        action_probs = self.model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, action_probs

    @ray.remote(num_cpus=1)
    def train(self, num_episodes, num_traces):
        
        for _ in range(num_episodes):
            rewards = deque()
            log_probs = []
            action_probs = []
            
            gradient = 0
            entropy_loss = 0
            # Generate num traces
            for _ in range(num_traces):
                state = self.env.reset()
                done = False
                # Sample a trace
                while not done:
                    action, log_prob, action_prob = self.select_action(state.flatten())
                    state, reward, done, _ = self.env.step(action)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    action_probs.append(action_prob)


                R = 0
                returns = deque()
                
        
                for r in reversed(rewards):
                    R = r + self.gamma * R
                    returns.appendleft(R)
                
                normalized_returns = torch.tensor(returns).to(self.device)
                normalized_returns = (normalized_returns - normalized_returns.mean()) / (normalized_returns.std())
                
                for log_prob, R in zip(log_probs, normalized_returns):
                    gradient += -log_prob * R

                for action_prob in action_probs:
                    entropy = -(action_prob * action_prob.log()).sum()
                    entropy_loss += entropy


            gradient = gradient + self.entropy_weight * entropy_loss

            self.optimizer.zero_grad()
            
            gradient.backward()
            self.optimizer.step()

            print('Sum reward:' + str(sum(rewards)))


if __name__ == '__main__':
    env = Catch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_episodes = 500
    traces_per_episode = 5
    agent = ReinforceAgent(env, device, 128)
    win_rates = agent.train(num_episodes, traces_per_episode)
    