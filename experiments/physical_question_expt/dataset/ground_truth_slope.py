import numpy as np

def compute_displacement(ball_radius, ball_mass, slope_angle_deg, g, obstacle_positions, t, obstacle_height=0.2):
    """
    Compute the displacement along the slope at time t, considering:
    - Rolling without slipping for a solid sphere.
    - Instant stop upon collision with triangular obstacles.
    
    Parameters:
    - ball_radius: Radius of the ball (m).
    - ball_mass: Mass of the ball (kg). (Not used directly in the acceleration formula for a solid sphere.)
    - slope_angle_deg: Incline angle (degrees).
    - g: Gravitational acceleration (m/s²).
    - obstacle_positions: List of distances (m) from the top where triangular apexes are placed.
    - t: Observation time (s).
    - obstacle_height: Height of each triangle (m), default 0.2 m.
    
    Returns:
    - s_true: Displacement along the slope (m) at time t, stopping early if a collision occurs.
    """
    R = ball_radius
    theta = np.deg2rad(slope_angle_deg)
    # acceleration for a solid sphere rolling without slipping
    a = (5/7) * g * np.sin(theta)
    # ideal displacement without obstacles
    s_ideal = 0.5 * a * t**2
    
    # compute collision distances along slope for each obstacle
    s_collision = []
    for d in obstacle_positions:
        delta = R - obstacle_height
        if delta < 0 or delta > R:
            # no possible contact geometry, ignore this obstacle
            continue
        s_c = d - np.sqrt(R**2 - delta**2)
        s_collision.append(s_c)
    
    # the ball stops at the earliest collision if any
    s_true = min([s_ideal] + s_collision) if s_collision else s_ideal
    return s_true

# Example usage:
if __name__ == "__main__":
    ball_radius = 0.1  # meters
    ball_mass = 1.0    # kg
    slope_angle_deg = 30
    g = 9.81
    obstacle_positions = [1.0, 2.5, 4.0]
    t = 2.0

    displacement = compute_displacement(
        ball_radius, ball_mass, slope_angle_deg, g, obstacle_positions, t
    )
    print(f"Displacement along slope at t={t}s: {displacement:.4f} m")

