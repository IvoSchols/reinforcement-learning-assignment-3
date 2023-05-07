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
def start_experiment(agent, num_repeats, num_episodes, num_traces, experiment_name, width, height, speed, observation_type="pixel"):
    wandb.init(
        # set the wandb project where this run will be logged
        project="RL3",
        
        # track hyperparameters and run metadata
        config={
            "part": 2,
            "lr": agent.learning_rate,
            "gamma": agent.gamma,
            "traces_per_episode": num_traces,
            "experiment_name": experiment_name,
            "observation_type": observation_type,
            "width": width,
            "height": height,
            "speed": speed,
        },
        group='part 2'
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
            "lr": 0.01,
            "gamma": 0.9,
            "traces_per_episode": 5,
            "main_station": True,
        },
        group='part 2'
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    num_repeats = 15
    num_episodes = 500
    traces_per_episode = 5

    hidden_size = 64
    learning_rate = 0.01
    gamma = 0.9
    
    #[width,height, speed] 
    tests = [(7,7,0.5),(7,14,0.5),(14,7,0.5),(14,14,0.5),
             (7,7,1),(7,14,1),(14,7,1),(14,14,1),
             (7,7,2),(7,14,2),(14,7,2),(14,14,2),
            ]


    futures = []
    


    #send lr experiments to ray
    for test in tests:
        width, height, speed = test

        env = Catch(width, height, speed)
        agent = ActorCriticFullAgent(env, device, hidden_size, learning_rate, gamma)
        
        win_rates = start_experiment.remote(agent, num_repeats, num_episodes, traces_per_episode, "diff envs", width, height, speed)

        futures.append(win_rates)



    #vector experiments
    env = Catch(speed=1 ,observation_type='vector')
    agent = ActorCriticFullAgent(env, device, hidden_size, learning_rate, gamma)
    futures.append(start_experiment.remote(agent, num_repeats, num_episodes, traces_per_episode, "diff envs", 7, 7, 1, "vector"))

    env = Catch(speed=2 ,observation_type='vector')
    agent = ActorCriticFullAgent(env, device, hidden_size, learning_rate, gamma)
    futures.append(start_experiment.remote(agent, num_repeats, num_episodes, traces_per_episode, "diff envs", 7, 7, 2, "vector"))
    


    results = ray.get(futures)

    wandb.finish()