import torch 

import numpy as np
import torch.nn as nn
from BaseAgent import Agent
from ActorCriticModel import ActroCriticModel
import sys



class ReinforceAgent(Agent):

    def __init__(self, env, device, optimizer='adam', lossFunction='huber', state_space= 98):
        super().__init__(env, device, optimizer, lossFunction)
        self.state_space = state_space
        self.num_steps = 10
        self.policy = ActroCriticModel(self.state_space)
        self.all_rewards = []
        self.all_lengths = []
        self.average_lengths = []



    def sample_traces(self, episode):
        rewards = []
        values = []
        log_probs = [] 
        for steps in range(self.num_steps):
            value, policy_dist = self.policy.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(self.state_space, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = self.env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            
            if done or steps == self.num_steps-1:
                Qval, _ = self.policy.forward(new_state)
                Qval = Qval.detach().numpy()[0,0]
                self.all_rewards.append(np.sum(rewards))
                self.all_lengths.append(steps)
                self.average_lengths.append(np.mean(self.all_lengths[-10:]))
                if episode % 10 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break

    def optimize_model(self, state, action, next_state, reward, done, rewards):
        

        return reward
    


