import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fixed parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0/3.0

# Time span and step size
t0, t1, dt = 0.0, 100.0, 0.01
t = np.arange(t0, t1 + dt, dt)

# Define the Lorenz system derivative
def lorenz(state, sigma, rho, beta):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        rho * x - y - x * z,
        x * y - beta * z
    ])

# Runge-Kutta 4 integrator
def rk4(initial_state):
    traj = np.zeros((len(t), 3))
    traj[0] = initial_state
    for i in range(len(t) - 1):
        k1 = lorenz(traj[i], sigma, rho, beta)
        k2 = lorenz(traj[i] + 0.5*dt*k1, sigma, rho, beta)
        k3 = lorenz(traj[i] + 0.5*dt*k2, sigma, rho, beta)
        k4 = lorenz(traj[i] + dt*k3, sigma, rho, beta)
        traj[i+1] = traj[i] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return traj

# Initial conditions for analysis
initial_conditions = {
    "w1 = (1,1,1)": np.array([1.0, 1.0, 1.0]),
    "w2 = (1,1,0)": np.array([1.0, 1.0, 0.0]),
    "w3 = (1,1,0.5)": np.array([1.0, 1.0, 0.5])
}

# Compute trajectories
trajectories = {name: rk4(state) for name, state in initial_conditions.items()}

# 3D plot of trajectories for different initial conditions
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
for name, traj in trajectories.items():
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label=name)
ax.set_title("Lorenz Attractor Trajectories for Different Initial Conditions")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.legend()
ax.grid(True)

# Time-series subplot comparison
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
components = ['X', 'Y', 'Z']
for idx, comp in enumerate(components):
    for name, traj in trajectories.items():
        axes[idx].plot(t, traj[:, idx], label=name)
    axes[idx].set_ylabel(f"{comp}(t)")
    axes[idx].legend(loc="upper right")
axes[-1].set_xlabel("Time")
fig.suptitle("Component Time Series for Different Initial Conditions")

# Compute distance between the first two trajectories
names = list(trajectories.keys())
diff = np.linalg.norm(trajectories[names[0]] - trajectories[names[1]], axis=1)

# Plot divergence over time
plt.figure(figsize=(6,4))
plt.semilogy(t, diff)
plt.title("Divergence of Two Close Initial Conditions (Log Scale)")
plt.xlabel("Time")
plt.ylabel("||w1 - w2||")
plt.grid(True)

plt.tight_layout()
plt.show()
