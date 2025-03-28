import os

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")

class LLM_Agent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def decide_move(self, state):
        prompt = f"""
        You are controlling a robot in a 2D physical environment with moving obstacles.
        Your goal is to avoid collisions while progressing toward the target location.

        Current robot state:
        - Position: {state['robot']['position']}
        - Velocity: {state['robot']['velocity']}

        Obstacles:
        {"".join([f"- Obstacle {i+1} at {obs['position']} moving at {obs['velocity']};" for i, obs in enumerate(state['cats'])])}

        Instead of choosing from discrete directions, choose an acceleration vector (ax, ay) to apply to the robot.

        Output the decision in this format:
        [{{
          "acceleration": [ax, ay],
          "duration": float(seconds),
        }},
        ...,
        {{
          "acceleration": [-ax, -ay],
          "duration": float(seconds),
        }} % as brake
        ] 
    
        Only return the JSON object with no explanation or markdown.
        """

        system_prompt = "You are an AI robot that avoids dynamic obstacles using acceleration control."
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            generated_answer = str(completion.choices[0].message.content)
            return generated_answer
        except Exception as e:
            return f"API error: {e}"

