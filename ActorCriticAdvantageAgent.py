
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ActorCriticModel import ActorCritic
from catch import Catch
import ray

class ActorCriticAdvantageAgent:
    def __init__(self, env, device, hidden_size=64, learning_rate=0.001, gamma=0.99, entropy_weight=0.01):
        self.env = env
        self.device = device
        self.num_inputs = np.prod(env.observation_space.shape)
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight

        self.model = ActorCritic(self.num_inputs, self.num_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def collect_traces(self, num_traces):
        traces = []
        state = self.env.reset()

        for _ in range(num_traces):
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs, _ = self.model(state_tensor)
                action = torch.multinomial(action_probs, 1).item()

                next_state, reward, done, _ = self.env.step(action)
                traces.append((state, action, reward, next_state, done))

                if done:
                    state = self.env.reset()
                else:
                    state = next_state

        return traces

    def compute_advantages(self, traces):
        advantages = []
        for t in range(len(traces)):
            state, _, reward, next_state, done = traces[t]
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            _, state_value = self.model(state_tensor)
            _, next_state_value = self.model(next_state_tensor)

            if done:
                next_state_value = torch.tensor(0.0)

            advantage = reward + self.gamma * next_state_value - state_value
            advantages.append(advantage)

        return advantages

    def update_policy(self, traces, advantages):
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for t, advantage in zip(traces, advantages):
            state, action, _, _, _ = t
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs, _ = self.model(state_tensor)

            critic_loss = advantage.pow(2)
            critic_losses.append(critic_loss)

            actor_loss = -torch.log(action_probs[0, action]) * advantage.detach()
            actor_losses.append(actor_loss)

            entropy_loss = -torch.sum(action_probs * torch.log(action_probs), dim=1)
            entropy_losses.append(entropy_loss)

        loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum() - self.entropy_weight * torch.stack(entropy_losses).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_winrate(self, traces):
        total_games = 0
        games_won = 0
        for trace in traces:
            _, _, reward, _, _ = trace
            total_games += (reward != 0)
            games_won += (reward == 1)

        return games_won / total_games

    @ray.remote(num_cpus=1)
    def train(self, num_episodes, num_traces):
        win_rates = []
        for episode in range(num_episodes):
            traces = self.collect_traces(num_traces)
            advantages = self.compute_advantages(traces)
            self.update_policy(traces, advantages)
            print(f'Episode {episode + 1}, Trace len: {len(traces)}')
            win_rates.append(self.get_winrate(traces))
        return win_rates
    
if __name__ == '__main__':
    env = Catch()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_episodes = 500
    traces_per_episode = 5
    agent = ActorCriticAdvantageAgent(env, device)
    win_rates = agent.train(num_episodes, traces_per_episode)
    