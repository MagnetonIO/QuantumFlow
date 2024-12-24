import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_phase_space(trajectory):
    q_vals, p_vals, H_vals = trajectory.T
    time = range(len(q_vals))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(q_vals, p_vals, H_vals, label='Phase Space Evolution')
    ax.set_xlabel("Position (q)")
    ax.set_ylabel("Momentum (p)")
    ax.set_zlabel("R^nH (Hamiltonian)")
    ax.set_title("Derived Hamiltonian Phase Space Visualization")
    plt.legend()
    plt.show()
