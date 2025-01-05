import numpy as np


class RSO:
    def __init__(self, x0, n_iter, n_actions, env):
        self.x0 = x0
        self.n_iter = n_iter
        self.n_actions = n_actions
        self.env = env

    def _evaluate(self, actions):
        self.env.reset()
        self.env.unwrapped.state = self.x0
        accumulated_reward = 0
        for action in actions:
            reward = self.env.step(np.array([action]))[1]
            accumulated_reward += reward
        return accumulated_reward

    def optimize(self):
        accumulated_rewards = np.zeros(self.n_iter)
        actions_vector = np.random.uniform(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            size=(self.n_iter, self.n_actions),
        )

        for i in range(self.n_iter):
            accumulated_rewards[i] = self._evaluate(actions_vector[i])
        
        actions = actions_vector[np.argmax(accumulated_rewards)]
        reward = np.max(accumulated_rewards)
        return actions, reward
