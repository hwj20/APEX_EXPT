
import base64
import json
import os
import re

import requests
import openai

api_key = ""

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
    print(decision)
    try:
        decision = decision.lower()
        decision = json.loads(decision)
        x = decision['x']
        y = decision['y']
        return (x,y,0.0)
    except Exception as e:
        raise ValueError("error parsing move")



class LLM_Agent:
    def __init__(self, model="gpt-4o"):
        self.model = model
      #  self.client = OpenAI(api_key=api_key)
        openai.api_key = api_key

    def decide_move_kid_apex(self,state,rolling_results):

        prompt = f"""
You are controlling a robot ball on 2D board. It can stop any object near in any movement
You can move the ball to a location (x,y) in 1 sec
Current_state:{state}
The green car is reaching the child in red T-shirt in 5 sec.

Physical Engine Result: {rolling_results}
Return your result as a JSON dictionary: {{"x": ..., "y": ...}} or {{"x":-99, "y":-99}} if you think no need of action
Return Only The JSON without Markdown
        """

        system_prompt = "You are controlling a robot ball"
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },

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
                return decode(strip_markdown(generated_answer))
            except ValueError as e:
                return "json error"
        except Exception as e:
            return f"API error: {e}"
    
    def decide_move_box_apex(self,state,rolling_results):
    #    return decode('{\"x\":0,\"y\":18}')

        prompt = f"""
You are controlling a robot arm in a 2D tabletop environment. Two balls are moving on the table: a red ball and a green ball. The red ball is stationary, and the green ball is moving toward it.

Your task is to **prevent a collision** between them by moving the robot arm to intercept the green ball.

Please choose a 2D target position (x, y), where the robot arm should go to block the green ball’s path. The robot arm will then move to the position (x, y, 0.5) in 3D space, hovering slightly above the table.

Make sure the chosen position is effective in preventing the collision, but also avoid placing the robot arm too close to the red ball.

Current_state:{state}
Physical Engine Result: {rolling_results}
Return your result as a JSON dictionary: {{"x": ..., "y": ...}} or {{"x":-99, "y":-99}} if you think no need of action
Return Only The JSON without Markdown
        """

        system_prompt = "You are controlling a robot arm in a 2D tabletop environment."
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },

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
                return decode(strip_markdown(generated_answer))
            except ValueError as e:
                return "json error"
        except Exception as e:
            return f"API error: {e}"
    
    def decide_move_box(self,state,available_pos):

        prompt = f"""
You are controlling a robot arm in a 2D tabletop environment. Two balls are moving on the table: a red ball and a green ball. The red ball is stationary, and the green ball is moving toward it.

Your task is to **prevent a collision** between them by moving the robot arm to intercept the green ball.

Please choose a 2D target position (x, y), where the robot arm should go to block the green ball’s path. The robot arm will then move to the position (x, y, 0.5) in 3D space, hovering slightly above the table.

Make sure the chosen position is effective in preventing the collision, but also avoid placing the robot arm too close to the red ball.
Curretn State {json.dumps(state, indent=2)}

Available Positions:{available_pos}

Return your result as a JSON dictionary: {{"x": ..., "y": ...}} or {{"x":-99, "y":-99}} if you think no need of action
Return Only The JSON without Markdown
"""

        system_prompt = "You are controlling a robot arm in a 2D tabletop environment."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },

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
            return decode(strip_markdown(generated_answer))

        except ValueError as e:
            print(e)
            return "json error"
        except Exception as e:
            print(e)
            return f"API error: {e}"

