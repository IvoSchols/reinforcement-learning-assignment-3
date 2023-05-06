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


class ReinforceModel(nn.Module):
    def __init__(self,state_space=98, action_space=3):
        super(ReinforceModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_space,128), 
            nn.ReLU(),
            nn.Linear(128,action_space),
            nn.Softmax(dim=0))
        
    def forward(self, x):
        return self.layers(x)



env = Catch()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



policy = ReinforceModel().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def select_action(state):
    state = torch.tensor(state).to(device)
    action_probs = policy(state)
    dist = Categorical(action_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


def main():
    gamma = 0.99
    render = False
    
    M = 50 # Number of traces generated for Monte Carlo
    converged = False

    while not converged:
        rewards = deque()
        log_probs = []
        
        gradient = 0

        # Generate M traces
        for _ in range(M):
            state = env.reset()
            done = False
            # Sample a trace
            while not done:
                action, log_prob = select_action(state.flatten())
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                if render:
                    env.render()


            R = 0
            returns = deque()
            
            for r in reversed(rewards):
                R = r + gamma * R
                returns.appendleft(R)
            
            normalized_returns = torch.tensor(returns).to(device)
            normalized_returns = (normalized_returns - normalized_returns.mean()) / (normalized_returns.std())
            
            for log_prob, R in zip(log_probs, normalized_returns):
                gradient += -log_prob * R

        optimizer.zero_grad()
        
        gradient.backward()
        optimizer.step()

        # Check for convergence
        sum_reward = sum(rewards)

        print('Sum reward:' + str(sum_reward))
        
        if sum_reward > 0:
            render = True

        if sum_reward > 100:
            converged = True


if __name__ == '__main__':
    main()
