import torch.nn as nn
import torch
from ActorCriticModel import ActorCritic
import torch.optim as optim
from catch import Catch


def collect_traces(env, model, num_traces, render):
    traces = []
    state = env.reset()

    for _ in range(num_traces):# n traces
        done = False
        while not done:#one trace
            if render:
                env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs, _ = model(state_tensor)
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, done, _ = env.step(action)
            traces.append((state, action, reward, next_state, done))

            if done:
                state = env.reset()
            else:
                state = next_state
    
    return traces

def compute_returns(traces, n_steps=5, gamma=0.99):
    returns = []
    for t in range(len(traces) - n_steps + 1):
        n_step_return = 0
        for i in range(n_steps):
            _, _, reward, _, done = traces[t + i]
            n_step_return += (gamma ** i) * reward
            if done:
                break

        returns.append(n_step_return)
    
    return returns



def update_policy(model, optimizer, traces, returns):
    actor_losses = []

    for t, ret in zip(traces, returns):
        state, action, _, _, _ = t
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs, _ = model(state_tensor)

        actor_loss = -torch.log(action_probs[0, action]) * ret
        actor_losses.append(actor_loss)

    loss = torch.stack(actor_losses).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(env, model, optimizer, num_episodes, num_traces):
    render = False
    for episode in range(num_episodes):
        traces = collect_traces(env, model, num_traces, render)
        returns = compute_returns(traces)
        update_policy(model, optimizer, traces, returns)
        print(f'Episode {episode + 1}, Trace len: {len(traces)}')
        if episode > 500:
            render = True

if __name__ == '__main__':
    env = Catch()
    num_inputs = 98
    num_actions = env.action_space.n
    hidden_size = 128

    model = ActorCritic(num_inputs, num_actions, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_episodes = 1000
    train(env, model, optimizer, num_episodes, 5)