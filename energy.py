import numpy as np

class Energy():
    def __init__(self, k, m, g, l):
        self.k = k
        self.m = m
        self.g = g
        self.l = l
        
    def update(self, theta, dtheta):
        ek = 0.5 * self.m * self.l**2 * dtheta**2
        ep = self.m * self.g * self.l * (1 - np.cos(theta + np.pi))
        energy = ek + ep
        
        energy_desired = 2 * self.m * self.g * self.l
        
        u = -self.k * (energy - energy_desired) * dtheta
        return u