import random
import json

random.seed(42)
def generate_prompt():
    # Random masses between 0.5 and 10.0 kg
    masses = [round(random.uniform(0.5, 10.0), 2) for _ in range(3)]
    # Random 2D positions between -10 and 10 m
    positions = [[round(random.uniform(-10.0, 10.0), 2) for _ in range(2)] for _ in range(3)]
    
    # Construct question string
    q = (
        "\n        Three bodies with masses {m1} kg, {m2} kg, and {m3} kg "
        "are located at positions ({x1}, {y1}) m, ({x2}, {y2}) m, and ({x3}, {y3}) m respectively. "
        "Using G = 6.674e-11 N·m²/kg², calculate the acceleration components (ax, ay) of each body at t = 1s.\n    ".format(
            m1=masses[0], m2=masses[1], m3=masses[2],
            x1=positions[0][0], y1=positions[0][1],
            x2=positions[1][0], y2=positions[1][1],
            x3=positions[2][0], y3=positions[2][1]
        )
    )
    
    return {
        "type": "Three-Body Motion",
        "question": q,
        "parameters": {
            "masses": masses,
            "positions": positions
        },
        "answer_json": {
            "acceleration_1_x": "",
            "acceleration_1_y": "",
            "acceleration_2_x": "",
            "acceleration_2_y": "",
            "acceleration_3_x": "",
            "acceleration_3_y": ""
        }
    }
if __name__ == '__main__':
    prompts = [generate_prompt() for _ in range(10)]
    with open('3_body_question.json', 'w') as f:
        json.dump(prompts, f, indent=4)
    print("Generated prompts.json with 10 three-body motion prompts.")

