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
        self.lr_critic = 0.001
        self.gamma = 0.99
        self.n = 5
        self.state_space = state_space
        self.action_dim = env.action_space.n
        self.policy = ActorCriticModel(self.state_space, self.action_dim).to(device)
        self.optimizer_actor = optim.Adam(self.policy.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = optim.Adam(self.policy.critic.parameters(), lr=self.lr_critic)

    def update_policy(self, traces):
        T = len(traces)
        
        critic_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        actor_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)

        for t in range(T):
            state_t, action_t, _, _, _ = traces[t]
            state_t_tensor = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0).to(self.device)
            # n-step target
            Q_n = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            for k in range(self.n):
                if t + k < T and t + self.n < T:
                    _, _, r_t_k, _, _ = traces[t + k]
                    r_t_k_tensor = torch.tensor(r_t_k, dtype=torch.float32)
                    state_t_n_tensor = torch.tensor(traces[t + self.n][0], dtype=torch.float32).unsqueeze(0).to(self.device)
                    Q_n += (self.gamma ** k) * (r_t_k_tensor + self.policy.critic(state_t_n_tensor.flatten()).item())

            # Descent value loss 
            critic_loss += (Q_n - self.policy.critic(state_t_tensor.flatten()).squeeze()) ** 2
            # Ascent pol grad 
            flatten = self.policy.actor(state_t_tensor.flatten(1)).gather(1, action_t.clone().detach().view(-1, 1))
            log_probs = torch.log(flatten)
            actor_loss += -Q_n * log_probs.squeeze()

        # Ascent policy gradient
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Ascent policy gradient
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).flatten().unsqueeze(0)
        action_probs = self.policy.actor(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action

    def run_episode(self, episode):
        state = self.env.reset()


        traces = []

        while True:
            action = self.select_action(state)

            next_state, reward, done, _ = self.env.step(action.item())

            traces.append((state, action, reward, next_state, done))

            
            if (episode > 500):
                self.env.render()
            if done:
                break
            
            state = next_state

        self.update_policy(traces)

        if episode % 50 == 0:
            print(f"Episode {episode}, Total episode length: {len(traces)}")


    def train(self, num_episodes=10000):
        for episode in range(num_episodes):
            self.run_episode(episode)



if __name__ == "__main__":
    env = Catch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = ActorCriticAgent(env, device)
    actor.train()