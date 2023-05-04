
import numpy as np
import torch

from BaseAgent import Agent
from ReinforceAgent import ReinforceAgent
from catch import Catch
from torch.distributions import Categorical

class Round():

    def __init__(self, agent: Agent, env: Catch, device):
        self.agent = agent # TODO: mount to GPU here or in agent?
        self.env = env
        self.device = device

    # Generate a full monte carlo trace
    def sample_trace(self, state):
        rewards = []
        log_probs = []
        done = False
        while not done:# and len(rewards) < steps:
            state = torch.tensor(state).flatten().to(self.device)
            action_probabilities = self.agent.select_action(state)
            action_distribution = Categorical(action_probabilities)

            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action)
            log_probs.append(log_prob)


            next_state, reward, done, _ = self.env.step(action.item())
            rewards.append(reward)
            state = next_state

        return rewards, log_probs



    def run(self):
        M = 1000 # Number of traces generated for Monte Carlo
        gamma = 0.99 # Discount factor
        eta = 0.01 # Learning rate

        # self.agent.net.randomize() # Randomize weights -> By default pytorch uses LeCun initialization (random from normal distr)
        converged = False

        while not converged:
            gradient = 0
            rewards = []


            for _ in range(M):
                state = self.env.reset()
                sample_rewards, sample_log_probs = self.sample_trace(state)
                rewards.append(sum(sample_rewards))

                R = 0
                
                for i in reversed(range(len(sample_rewards))):
                    R = sample_rewards[i] + gamma * R
                    gradient += R * sample_log_probs[i]
                
  
            loss = -1 * (gradient / M) * eta
            self.agent.optimizer.zero_grad()
            loss.backward()
            self.agent.optimizer.step()

            # Check for convergence
            skill = sum(sample_rewards) / len(sample_rewards)
            if skill > 0.9:
                    converged = True
            
            print(skill)

            if skill > 0.6:
                self.env.render()


        return rewards


if __name__ == "__main__":
    env = Catch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    agent = ReinforceAgent(env, device, 'adam', 'huber', np.prod(env.observation_space.shape), env.action_space.n)
    round = Round(agent, env, device)
    round.run()
    
    
