from catch import Catch
import torch
from ActorCriticAdvantageAgent import ActorCriticAdvantageAgent
from ActorCriticBootstrapAgent import ActorCriticBootstrapAgent
from ActorCriticFullAgent import ActorCriticFullAgent

import numpy as np
import ray


def start_experiment(agent, num_repeats, num_episodes, num_traces):
    futures = [agent.train.remote(agent, num_episodes, num_traces) for i in range(num_repeats)]
    all_winrates = ray.get(futures) 
    averages = np.array(all_winrates).mean(axis=0)
    return averages



if __name__ == '__main__':
    ray.init(num_cpus=8)
    env = Catch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    num_repeats = 15
    num_episodes = 500
    traces_per_episode = 5

    hidden_size = 64
    learning_rate = 0.001
    gamma = 0.99

    agent = ActorCriticAdvantageAgent(env, device, hidden_size, learning_rate, gamma)
    win_rates = start_experiment(agent, num_repeats, num_episodes, traces_per_episode)
    print(win_rates)

