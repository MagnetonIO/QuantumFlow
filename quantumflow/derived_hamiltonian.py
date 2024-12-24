import numpy as np

class DerivedHamiltonian:
    def __init__(self, q0, p0, g, order=3):
        self.q = q0
        self.p = p0
        self.g = np.array(g)
        self.order = order
        self.hbar = 1.0
        self.mass = 1.0
        self.k = 1.0
        self.alpha = [1.0, 0.5, 0.1]  # Higher-order coefficients

    def H0(self, q, p):
        return (p**2) / (2 * self.mass) + 0.5 * self.k * q**2

    def H1(self, q):
        return (self.hbar * self.k) / 2

    def Hn(self, q, n):
        return self.alpha[n - 1] * self.hbar**n * q**n

    def RnH(self, q, p):
        H = self.H0(q, p) + self.H1(q)
        for i in range(2, self.order + 1):
            H += self.Hn(q, i)
        return H

    def evolve(self, steps=500, dt=0.01):
        trajectory = []
        for _ in range(steps):
            p_half = self.p - 0.5 * dt * self.q
            self.q += dt * p_half
            self.p = p_half - 0.5 * dt * self.q
            H = self.RnH(self.q, self.p)
            trajectory.append((self.q, self.p, H))
        return np.array(trajectory)
