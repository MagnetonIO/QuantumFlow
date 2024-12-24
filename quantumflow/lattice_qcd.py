import jax.numpy as jnp

class LatticeQCD:
    def __init__(self, lattice_size):
        self.lattice_size = lattice_size
        self.gauge_field = jnp.ones((lattice_size, lattice_size))

    def evolve(self, steps=100):
        for _ in range(steps):
            self.gauge_field += 0.1 * jnp.sin(self.gauge_field)

    def compute_action(self):
        return jnp.sum(self.gauge_field**2)
