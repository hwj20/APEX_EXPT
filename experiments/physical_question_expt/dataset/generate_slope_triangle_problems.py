import random
import json

def generate_triangle_collision_prompt():
    """
    Generate a refined slope task where a spherical ball rolls down and hits triangular obstacles.

    Returns:
        dict: Prompt dictionary with physical setup and explicit reasoning target.
    """
    slope_length = round(random.uniform(5.0, 10.0), 2)
    slope_angle_deg = random.choice([15, 20, 25])  # Gentle slope
    ball_radius = 0.1  # m
    ball_mass = 0.5  # kg
    gravity = 9.81  # m/s^2

    obstacle_count = random.randint(1, 2)
    obstacle_positions = sorted([
        round(random.uniform(0.5, slope_length - 0.5), 2)
        for _ in range(obstacle_count)
    ])
    obstacle_height = 0.2  # triangular obstacles height in meters
    observation_time = 1.0  # second

    question = (
        f"A spherical ball of radius {ball_radius} m and mass {ball_mass} kg is placed at the top of an inclined plane "
        f"{slope_length} meters long, angled at {slope_angle_deg} degrees from the horizontal. The ball rolls down without slipping under "
        f"gravity (g = {gravity} m/s²).\n\n"

        f"There are {obstacle_count} identical triangular obstacles rigidly attached to the slope (i.e., fixed and immovable), each forming an isosceles triangle "
        f"with height 0.2 m and base 0.4 m. Each triangle is placed such that:\n"
        f" - Its base lies flush against the slope surface (i.e., fully adheres to the incline), and\n"
        f" - Its apex points *normal* to the slope surface (i.e., orthogonal to the inclined plane, not vertically upward in global coordinates).\n\n"

        f"The apex of each triangle is located at distances {obstacle_positions} meters along the slope, measured from the top (release point).\n"
        f"Assume that the ball stops instantly if the vertical distance from the center of the ball to any apex becomes less than or equal to the ball's radius.\n\n"

        f"At exactly t = {observation_time} seconds after release, where is the center of the ball along the slope?\n"
        f"Give a precise numeric answer in meters along the slope."
    )

    return {
        "type": "Triangle Collision Prediction",
        "question": question,
        "parameters": {
            "slope_length": slope_length,
            "slope_angle_deg": slope_angle_deg,
            "ball_radius": ball_radius,
            "ball_mass": ball_mass,
            "gravity": gravity,
            "obstacle_positions": obstacle_positions,
            "observation_time": observation_time
        },
        "answer_json": {
            "ball_center_position_at_1s": ""
        }
    }

if __name__ == "__main__":
    prompts = [ generate_triangle_collision_prompt() for _ in range(10)]
    with open("slope_branch_question.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)
    print("Generated slope_branch_question.json with 10 slope branch interference prompts.")
