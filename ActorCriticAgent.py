import torch
from ActorCriticModel import ActorCriticModel
import torch.optim as optim
from catch import Catch
import numpy as np

class ActorCriticAgent():
    def __init__(self, env, device, state_space = 98):
        self.env = env
        self.device = device
        self.lr_actor = 0.001
        self.gamma = 0.99
        self.state_space = state_space
        self.action_dim = env.action_space.n
        self.policy = ActorCriticModel(self.state_space, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.actor.parameters(), lr=self.lr_actor)


    def update_policy(self, states, actions, rewards):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        #bootstrapping
        action_probs, values = self.policy(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        advantages = rewards - values.squeeze()

        
        actor_loss = (-action_log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss



        self.policy.zero_grad()
        loss.backward()
        self.optimizer.step()


    def run_episode(self, episode):
        state = self.env.reset()
        states, actions, rewards = [], [], []

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).flatten().unsqueeze(0)
            action_probs = self.policy.actor(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done, _ = self.env.step(action.item())

            states.append(state.flatten())
            actions.append(action)
            rewards.append(reward)

            # if (episode > 200):
            #     self.env.render()
            if done:
                break
            
            state = next_state



            
        # Compute cumulative discounted rewards
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0
        for t in reversed(range(len(rewards))):
            cumulative_reward = rewards[t] + self.gamma * cumulative_reward
            discounted_rewards[t] = cumulative_reward

        self.update_policy(states, actions, discounted_rewards)

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {len(rewards)}")


    def train(self, num_episodes=10000):
        for episode in range(num_episodes):
            self.run_episode(episode)



if __name__ == "__main__":
    env = Catch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = ActorCriticAgent(env, device)
    actor.train()