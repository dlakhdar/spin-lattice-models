import matplotlib.pyplot as plt
import numpy as np

up = "\u2191"  # ↑
down = "\u2193"  # ↑

boundary_conditions = {
    "periodic",
    "aperiodic",
    "buffered",
    "random",
}

# TODO: add glass
magnetic_character = {
    "ferromagnetic": 1,
    "antiferromagnetic": 1,
    "paramagnetic": 0,
}


def hamiltonian(lattice: "SpinLattice") -> float | int:
    interaction_energy = 0.0
    h_energy = np.sum(-lattice.spins * lattice.h)
    for i in range(1, lattice.N + 1):
        for j in range(1, lattice.N + 1):
            spin = lattice.spins[i, j]
            neighbors = np.array(
                [
                    lattice.spins[i - 1, j],
                    lattice.spins[i + 1, j],
                    lattice.spins[i, j - 1],
                    lattice.spins[i, j + 1],
                ]
            )

        interaction_energy += np.sum(-lattice.J * neighbors * spin)

    H = -lattice.J * interaction_energy - h_energy
    return H


def monte_carlo(lattice: "SpinLattice", iterations: (int)) -> list:
    energies = []

    print(lattice)
    for i in range(0, iterations + 1):
        index = np.random.choice(
            np.arange(1, lattice.N + 1), lattice.dimension, replace=False
        )
        E0 = hamiltonian(lattice)
        lattice.spins[*index] = -1.0 * lattice.spins[*index]
        E = hamiltonian(lattice)
        ΔE = E - E0
        if ΔE < 0:
            pass
        else:
            p_return = np.exp(-lattice.β * ΔE)
            accept_reject = np.random.choice([-1.0, 1.0],
                                             p=[p_return, 1 - p_return])
            lattice.spins[*index] = accept_reject * lattice.spins[*index]
        energies.append(E)

    return energies


# TODO: add dimension
class SpinLattice:
    def __init__(
        self,
        N: (int | float),
        dimension: (int) = 2,
        temperature: (int | float) = 1,
        boundary_condition: str = "buffered",
        spin_number: (int | float) = 1 / 2,
        scale_number: (int | float) = 2,
        buffer_pad: (int | float) = 0,
        interaction_strength: (int | float) = 1.0,
        magnetic_strength: (int | float) = 1.0,
        seed=1917,
    ):
        self.inverse_temperature = 1 / temperature
        self.β = 1 / temperature
        self.dimension = dimension
        self.d = dimension
        self.N = N
        final_value = (spin_number + 0.5 if
                       spin_number % 1 != 0 else spin_number + 1)
        spin_values = np.arange(-spin_number, final_value) * scale_number
        self.spins = np.random.choice(spin_values, size=(self.N, self.N))
        self.h = magnetic_strength
        self.bc = boundary_condition
        self.J = interaction_strength
        self.interaction_strength = interaction_strength

        if self.bc == "buffered":
            # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            self.spins = np.pad(self.spins, pad_width=1)

    def __str__(self):
        spin_string = np.where(
            self.spins == 1, up, np.where(self.spins == -1, down, self.spins)
        )
        return str(spin_string[1: self.N + 1, 1: self.N + 1])


if __name__ == "__main__":
    iterations = 10000
    lattice = SpinLattice(N=100, interaction_strength=1.0,
                          magnetic_strength=1.0)
    energies = monte_carlo(lattice, iterations)
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(iterations + 1), energies, color="orange")
    plt.plot(energies)
    plt.grid()
    plt.xlabel("step (time)")
    plt.ylabel("H (energy)")
    plt.title(
        "Metropolis-Hastings Applied to 2-D Ising Model: N=100,\
              J=1.0,h=1.0"
    )
