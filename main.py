from catch import Catch
import torch
from ActorCriticAdvantageAgent import ActorCriticAdvantageAgent
from ActorCriticBootstrapAgent import ActorCriticBootstrapAgent
from ActorCriticFullAgent import ActorCriticFullAgent
from Reinforce import ReinforceAgent

import numpy as np
import ray
import matplotlib
import matplotlib.pyplot as plt
import wandb

@ray.remote
def start_experiment(agent, num_repeats, num_episodes, num_traces, agent_name, experiment_name):
    wandb.init(
        # set the wandb project where this run will be logged
        project="RL3",
        
        # track hyperparameters and run metadata
        config={
            "agent": agent_name,
            "part": 1,
            "lr": agent.learning_rate,
            "gamma": agent.gamma,
            "traces_per_episode": num_traces,
            "experiment_name": experiment_name,
        },
        group='part 1'
    )
    agent_ref = ray.put(agent)
    futures = [agent.train.remote(agent_ref, num_episodes, num_traces) for i in range(num_repeats)]
    all_winrates = ray.get(futures) 
    averages = np.array(all_winrates).mean(axis=0)
    for avg in averages:
        wandb.log({
            "win_rate": avg
        })
    wandb.finish()
    return averages



if __name__ == '__main__':
    ray.init(num_cpus=8)
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="RL3",
        
        # track hyperparameters and run metadata
        config={
            "part": 1,
            "lr": 0.001,
            "gamma": 0.99,
            "traces_per_episode": 5
        },
        group='part 1'
    )
    env = Catch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    num_repeats = 15
    num_episodes = 500
    traces_per_episode = 5

    hidden_size = 64
    learning_rate = 0.001
    gamma = 0.99
    
    lr_to_test = [0.01, 0.001, 0.0001]
    gamma_to_test = [0.9, 0.99, 0.999]

    futures = []


    #send lr experiments to ray
    for lr in lr_to_test:
        bootstrapAgent = ActorCriticBootstrapAgent(env, device, hidden_size, lr, gamma)
        advantageAgent = ActorCriticAdvantageAgent(env, device, hidden_size, lr, gamma)
        fullAgent = ActorCriticFullAgent(env, device, hidden_size, lr, gamma)
        reinforceAgent = ReinforceAgent(env, device, hidden_size, lr, gamma)
        
        bootstrapAgent_win_rates = start_experiment.remote(bootstrapAgent, num_repeats, num_episodes, traces_per_episode, "AC with bootstrapping", "lr tune")
        advantageAgent_win_rates = start_experiment.remote(advantageAgent, num_repeats, num_episodes, traces_per_episode, "AC with basline subtraction", "lr tune")
        fullAgent_win_rates = start_experiment.remote(fullAgent, num_repeats, num_episodes, traces_per_episode, "AC with bootstrapping and basline subtraction", "lr tune")
        reinforceAgent_win_rates = start_experiment.remote(fullAgent, num_repeats, num_episodes, traces_per_episode, "Reinforce", "lr tune")


        futures.append(bootstrapAgent_win_rates)
        futures.append(advantageAgent_win_rates)
        futures.append(fullAgent_win_rates)
        futures.append(reinforceAgent_win_rates)


    for gamma_to_test in gamma_to_test:
        bootstrapAgent = ActorCriticBootstrapAgent(env, device, hidden_size, learning_rate, gamma_to_test)
        advantageAgent = ActorCriticAdvantageAgent(env, device, hidden_size, learning_rate, gamma_to_test)
        fullAgent = ActorCriticFullAgent(env, device, hidden_size, learning_rate, gamma_to_test)
        reinforceAgent = ReinforceAgent(env, device, hidden_size, learning_rate, gamma_to_test)
        

        bootstrapAgent_win_rates = start_experiment.remote(bootstrapAgent, num_repeats, num_episodes, traces_per_episode, "AC with bootstrapping", "gamma tune")
        advantageAgent_win_rates = start_experiment.remote(advantageAgent, num_repeats, num_episodes, traces_per_episode, "AC with basline subtraction", "gamma tune")
        fullAgent_win_rates = start_experiment.remote(fullAgent, num_repeats, num_episodes, traces_per_episode, "AC with bootstrapping and basline subtraction", "gamma tune")
        reinforceAgent_win_rates = start_experiment.remote(fullAgent, num_repeats, num_episodes, traces_per_episode, "Reinforce", "lr tune")


        futures.append(bootstrapAgent_win_rates)
        futures.append(advantageAgent_win_rates)
        futures.append(fullAgent_win_rates)
        futures.append(reinforceAgent_win_rates)


    results = ray.get(futures)
