
from BaseAgent import Agent
from catch import Catch


class Round():

    def __init__(self, agent: Agent, env: Catch):
        self.agent = agent # TODO: mount to GPU here or in agent?
        self.env = env

    def sample_trace(self, state):
        states = [state]
        actions = []
        rewards = []
        done = False

        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return states, actions, rewards



    def run(self):
        M = 1000 # Number of traces generated for Monte Carlo
        eta = 0.1 # Learning rate

        self.agent.net.randomize() # Randomize weights
        state = self.env.reset()
        converged = False
        rewards = []
        while not converged:
            gradient = 0
            for m in range(M):
                states, actions, rewards = self.sample_trace(state)
                

                cumultative_reward = 0

                for i in reversed(range(len(rewards))):
                    cumultative_reward = cumultative_reward * self.agent.gamma + rewards[i]
                    log_probs = self.agent.net(states[i])
                    gradient += cumultative_reward * log_probs


            # Update weights
            self.agent.optimize_model(eta * gradient)

            # Check for convergence
            if len(rewards) > 100:
                if sum(rewards[-100:]) / 100 > 0.9:
                    converged = True
            
            pass

    