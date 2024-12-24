import jax.numpy as jnp
from jax import grad

class TensorHamiltonian:
    def __init__(self, q, p):
        self.q = jnp.array(q)
        self.p = jnp.array(p)

    def H(self, q, p):
        return 0.5 * (p**2 + q**2)  # Harmonic Oscillator Example

    def evolve(self, steps=500, dt=0.01):
        for _ in range(steps):
            dq_dt = grad(self.H, argnums=1)(self.q, self.p)
            dp_dt = -grad(self.H, argnums=0)(self.q, self.p)
            self.q += dt * dq_dt
            self.p += dt * dp_dt
        return self.q, self.p
