
import numpy as np
from BaseAgent import Agent
from ReinforceAgent import ReinforceAgent
from catch import Catch


class Round():

    def __init__(self, agent: Agent, env: Catch):
        self.agent = agent # TODO: mount to GPU here or in agent?
        self.env = env

    def sample_trace(self, state):
        states = [state]
        actions = []
        rewards = []
        log_probs = []
        done = False

        while not done:
            action, log_prob = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

        return rewards, log_probs



    def run(self):
        M = 1000 # Number of traces generated for Monte Carlo
        eta = 1 # Learning rate

        # self.agent.net.randomize() # Randomize weights -> By default pytorch uses LeCun initialization (random from normal distr)
        converged = False
        all_rewards = []
        discounted_rewards = []
        while not converged:
            if True:
                self.env.render()

            gradient = 0
            state = self.env.reset()

            for m in range(M):
                sample_rewards, sample_log_probs = self.sample_trace(state)
                all_rewards.append(sum(sample_rewards))
                

                cumultative_reward = 0

                for i in reversed(range(len(sample_rewards))):
                    cumultative_reward = cumultative_reward * self.agent.gamma + sample_rewards[i]
                    log_probs = sample_log_probs[i]
                    gradient += cumultative_reward * log_probs
                
                discounted_rewards.append(cumultative_reward)

            # Update weights
            discounted_rewards = np.array(discounted_rewards)
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            gradient *= discounted_rewards

            self.agent.net.zero_grad()
            gradient.sum().backward()
            self.agent.optimizer.step()

            # Check for convergence
            if sum(all_rewards[-100:]) / 100 > 0.9:
                    converged = True
            
            print(sum(all_rewards[-100:]) / 100)

        return all_rewards


if __name__ == "__main__":
    env = Catch()
    agent = ReinforceAgent(env, 'cpu', 'adam', 'huber', state_space=98, action_space=env.action_space.n)
    round = Round(agent, env)
    round.run()
    
    
