import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fixed parameters
sigma = 10.0
beta = 8.0 / 3.0
initial_state = np.array([1.0, 1.0, 0.5])
t0, t1, dt = 0.0, 100.0, 0.01
t = np.arange(t0, t1 + dt, dt)

# Lorenz derivative and RK4 integrator
def lorenz(state, sigma, rho, beta):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        rho * x - y - x * z,
        x * y - beta * z
    ])

def rk4(initial_state, sigma, rho, beta, t):
    traj = np.zeros((len(t), 3))
    traj[0] = initial_state
    for i in range(len(t) - 1):
        k1 = lorenz(traj[i], sigma, rho, beta)
        k2 = lorenz(traj[i] + 0.5*dt*k1, sigma, rho, beta)
        k3 = lorenz(traj[i] + 0.5*dt*k2, sigma, rho, beta)
        k4 = lorenz(traj[i] + dt*k3, sigma, rho, beta)
        traj[i+1] = traj[i] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return traj

# Parameter scenarios
scenarios = {
    'Fixed Point (ρ=0.28)': 0.28,
    'Limit Cycle (Hopf, ρ=23.935)': 23.935,
    'Period-Doubling (ρ=28)': 28.0,
    'Chaos (ρ=35)': 35.0
}

# Simulate and plot
fig = plt.figure(figsize=(12, 10))
for idx, (title, rho) in enumerate(scenarios.items(), 1):
    traj = rk4(initial_state, sigma, rho, beta, t)
    # 3D phase
    ax = fig.add_subplot(4, 2, 2*idx-1, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], linewidth=0.5)
    ax.set_title(f"{title} - 3D Phase")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.grid(True)
    # Time series zoom on last cycles for clarity
    ax2 = fig.add_subplot(4, 2, 2*idx)
    window = slice(-2000, -1)  # last 20 seconds
    ax2.plot(t[window], traj[window, 0], label='X')
    ax2.set_title(f"{title} - X(t) Time Series (last 20s)")
    ax2.set_xlabel("Time"); ax2.set_ylabel("X")
    ax2.grid(True)

plt.tight_layout()
plt.show()
