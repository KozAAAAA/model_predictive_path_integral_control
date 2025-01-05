import numpy as np
import gymnasium as gym


class MPPI:
    def __init__(
        self,
        horizon,
        trajectories,
        temperature,
        noise_cov,
        iterations,
        env_options={},
    ):
        self.horizon = horizon
        self.trajectories = trajectories
        self.temperature = temperature
        self.noise_cov = noise_cov
        self.iterations = iterations
        self.env = gym.make(**env_options)

    def optimize(self, x, u_prev):
        u_optim = np.roll(u_prev, -1, axis=0)
        u_optim[-1] = np.zeros(self.env.action_space.shape[0])

        for i in range(self.iterations):
            effective_perturbations = np.zeros(
                (self.trajectories, self.horizon, self.env.action_space.shape[0])
            )
            weights = np.zeros(self.trajectories)
            for n in range(self.trajectories):
                perturbation = np.random.multivariate_normal(
                    mean=np.zeros(self.noise_cov.shape[0]),
                    cov=self.noise_cov,
                    size=self.horizon,
                )
                u = u_optim + perturbation
                clamped_u = np.clip(
                    u,
                    self.env.action_space.low,
                    self.env.action_space.high,
                )

                cost = self._evaluate(x, clamped_u)

                weights[n] = np.exp(-cost / self.temperature)
                effective_perturbations[n] = clamped_u - u_optim

            if np.sum(weights) == 0:
                weights = np.ones(self.trajectories)
            else:
                weights /= np.sum(weights)

            u_optim += np.sum(effective_perturbations * weights[:, None, None], axis=0)

        return u_optim

    def _evaluate(self, x, u):
        self.env.reset()
        self.env.unwrapped.state = x
        accumulated_cost = 0
        for action in u:
            reward = self.env.step(action)[1]
            cost = -reward
            accumulated_cost += cost
        return accumulated_cost
