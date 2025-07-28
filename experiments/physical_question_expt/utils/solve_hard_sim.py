import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Gravitational constant (set to 1 for normalized units)
G = 1.0

def deriv(t, y, m):
    """
    Compute derivatives for the three-body problem.

    Parameters:
    - t: float, current time (unused since system is time-invariant)
    - y: array_like, shape (12,): [x1, y1, x2, y2, x3, y3,
                                    vx1, vy1, vx2, vy2, vx3, vy3]
    - m: array_like, shape (3,): masses of the three bodies

    Returns:
    - dydt: ndarray, shape (12,): time derivatives [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]
    """
    # Positions
    r1 = y[0:2]
    r2 = y[2:4]
    r3 = y[4:6]
    # Velocities
    v1 = y[6:8]
    v2 = y[8:10]
    v3 = y[10:12]

    def acceleration(ri, rj, mj):
        diff = rj - ri
        dist3 = np.linalg.norm(diff)**3 + 1e-9  # softening to avoid singularity
        return G * mj * diff / dist3

    # Compute accelerations
    a1 = acceleration(r1, r2, m[1]) + acceleration(r1, r3, m[2])
    a2 = acceleration(r2, r1, m[0]) + acceleration(r2, r3, m[2])
    a3 = acceleration(r3, r1, m[0]) + acceleration(r3, r2, m[1])

    return np.concatenate((v1, v2, v3, a1, a2, a3))


def simulate(m, y0, t_span, dt=0.01):
    """
    Simulate the three-body system.

    Parameters:
    - m: array_like, masses of the three bodies
    - y0: array_like, initial state vector (length 12)
    - t_span: tuple, (t_start, t_end)
    - dt: float, time step for evaluation grid

    Returns:
    - t: ndarray of times
    - y: ndarray of states over time, shape (12, len(t))
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(fun=lambda t, y: deriv(t, y, m), t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-9)
    return sol.t, sol.y


def plot_trajectories(t, y):
    """
    Plot the trajectories of the three bodies.
    """
    plt.figure(figsize=(8, 8))
    plt.plot(y[0], y[1], '-', label='Body 1')
    plt.plot(y[2], y[3], '-', label='Body 2')
    plt.plot(y[4], y[5], '-', label='Body 3')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Three-Body Problem Trajectories')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Masses of the three bodies
    m = np.array([1.0, 1.0, 1.0])
    
    # Initial positions and velocities: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
    y0 = np.array([
        -1.0, 0.0,   # Body 1 position
         1.0, 0.0,   # Body 2 position
         0.0, 0.0,   # Body 3 position (center)
        
         0.0, 0.3,   # Body 1 velocity
         0.0, -0.3,  # Body 2 velocity
         0.0, 0.0    # Body 3 velocity
    ])

    # Time span for simulation
    t_span = (0.0, 20.0)
    dt = 0.01

    # Run simulation and plot
    t, y = simulate(m, y0, t_span, dt)
    plot_trajectories(t, y)
