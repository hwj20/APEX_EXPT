import base64
import json
import os
import re

import requests
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")


def strip_markdown(text: str) -> str:
    text = re.sub(r"```", "", text)
    text = re.sub("json", "", text)

    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# Control Policy
def decode(decision: str):
    try:
        decision = decision.lower()
        decision = json.loads(decision)

        # stay
        vel = [0.0, 0.0, 0.0]  # x, y, z
        duration = decision['duration']  # s

        if "left" in decision['move']:
            vel[0] = -3.0
        elif "right" in decision['move']:
            vel[0] = 3.0
        elif "up" in decision['move']:
            vel[1] = -3.0
        elif "down" in decision['move']:
            vel[1] = 3.0
        elif "jump" in decision['move']:
            vel[2] = 3.0  # jump
            duration = 0.1

        return {"velocity": vel, "duration": duration}
    except Exception as e:
        raise ValueError("error parsing move")


class LLM_Agent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def decide_move_apex(self, state, summary, available_move, apex_results):
        prompt = f"""
        You are controlling a robot in a 3D physical environment with moving obstacles.
        Your goal is to avoid collisions with cats while progressing toward the target location.

        Current state (The map has square walls located at x = ±5 meters and y = ±5 meters):
        {state}

        Obstacles:
        {summary}
        
        Available Moves:
        {available_move}
        
        Physical Engine Analysis:
        {apex_results}

        Output the decision in this format:
        {{
        "move": "stay",
        "duration": 1.0,
        }}
          
        Only return the JSON object with no explanation or markdown.
        """

        system_prompt = "You are an AI robot that avoids dynamic obstacles."
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            generated_answer = str(completion.choices[0].message.content)
            print(generated_answer)
            return decode(strip_markdown(generated_answer)), True
        except ValueError as e:
            return "json error", False
        except Exception as e:
            return f"API error: {e}", False

    def decide_move(self, state, available_move):
        prompt = f"""
        You are controlling a robot in a 3D physical environment with moving obstacles.
        Your goal is to avoid collisions with cats while progressing toward the target location.

        Current state (The map has square walls located at x = ±5 meters and y = ±5 meters):
        {state}

        Available Moves:
        {available_move}

        Output the decision in this format:
        {{
        "move": "stay",
        "duration": 1.0,
        }}

    
        Only return the JSON object with no explanation or markdown.
        """

        system_prompt = "You are an AI robot that avoids dynamic obstacles."
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            generated_answer = str(completion.choices[0].message.content)
            print(generated_answer)
            return decode(strip_markdown(generated_answer)), True
        except ValueError as e:
            return "json error", False
        except Exception as e:
            return f"API error: {e}", False

    def decide_move_vlm(self,state, available_move, image_path):
        prompt = f"""
        You are controlling a robot in a 3D physical environment with moving obstacles.
        Your goal is to avoid collisions with cats while progressing toward the target location.

        Current state (The map has square walls located at x = ±5 meters and y = ±5 meters):
        {state}

        Available Moves:
        {available_move}

        Output the decision in this format:
        {{
        "move": "stay",
        "duration": 1.0,
        }}
    
        Only return the JSON object with no explanation or markdown.
        
        Here is the screenshot(Red balls cat, green ball-your controlled agent):
        """

        system_prompt = "You are an AI robot that avoids dynamic obstacles."

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response_data = response.json()

            generated_answer = response_data["choices"][0]["message"]["content"]
            print(generated_answer)
            return decode(strip_markdown(generated_answer)), True
        except ValueError as e:
            return "json error", False
        except Exception as e:
            return f"API error: {e}", False
