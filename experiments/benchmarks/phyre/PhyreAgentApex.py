import base64
import json
import os
import re

import requests
from llm_router import *

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
def decode(clean):
    try:
        action_objs = json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', clean, re.DOTALL)
        if not m:
            raise ValueError("No valid JSON array found in decision.")
        try:
            action_objs = json.loads(m.group())
        except Exception as e:
            raise ValueError(f"Failed to parse JSON array: {e}")

    if not isinstance(action_objs, list):
        raise ValueError("Decoded JSON is not a list of actions.")

    res = []
    for obj in action_objs:
        try:
            x = float(obj['x'])
            y = float(obj['y'])
            r = float(obj['r'])
        except Exception as e:
            raise ValueError(f"Action object missing or invalid keys: {e}")
        res.append([x, y, r])

    return res



class LLM_Agent:
    def __init__(self, model="gpt-4.1"):
        self.model = model
        # self.client = OpenAI(api_key=api_key)

    def decide_move_apex(self, state):
        physical_engine_analysis = state.pop("Physical Engine Tested  Points", [])

        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        physical_engine_json = json.dumps(physical_engine_analysis, ensure_ascii=False, indent=2)

        prompt = f"""
You are given an initial scene containing various static objects: balls, bars, jars, and other structures.  
Each object has a known position, size, shape, and color.

You are allowed to perform a single action: place one red ball (not green) into the scene before the simulation starts.


Current state:
```json
{state_json}
```
Your goal:
Ensure that during the simulation, the green ball comes into contact with either a purple or blue object and remains in contact for at least 3 seconds.

Gravity points downward (−y), and all physical interactions follow Newtonian dynamics.

Notes:

Coordinates of objects are given by (x, y), normalized between 0 and 1,
where (0,0) is bottom-left and (1,1) is top-right of a 256×256 scene.

All objects fall under gravity except static ones (black and purple).
The dynamic flag in the object dict indicates if it can move.

During the simulation, objects move and collide according to Newtonian rules.

Physical Engine Tested  Points are some examples of engine-verified successful placements:
{physical_engine_analysis}

Note: This is a numerically sensitive task. To ensure reliable success under simulation, please consider selecting placements based on the verified engine-tested examples below. While creative reasoning is welcome, maintaining physical correctness is critical in this setting.

Action format:
You must return a single JSON array containing one or more “attempt” objects(suggested more tries).
Each object must have exactly these keys:

[
  {{
    "reason": "Why you chose this placement",
    "x": 0.123,   // normalized between 0 and 1
    "y": 0.456,   // normalized between 0 and 1
    "r": 0.03     // radius, e.g. 0.03
  }}
]
Only return the JSON array—no extra explanation, markdown, or text.
"""


        system_prompt = """You are a physics reasoning agent in a 2D environment governed by gravity."""
        try:
            messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
            ]
            generated_answer = call_llm(self.model,messages)
            print(generated_answer)
            return generated_answer
        except ValueError as e:
            return "json error"
        except Exception as e:
            return f"API error: {e}"

    def decide_move(self, state, available_move=[]):
        state_json = json.dumps(state, ensure_ascii=False, indent=2)

        prompt = f"""
You are given an initial scene containing various static objects: balls, bars, jars, and other structures.  
Each object has a known position, size, shape, and color.

You are allowed to perform a single action: place one red ball (not green) into the scene before the simulation starts.

Current state:
```json
{state_json}
```
Your goal:
Ensure that during the simulation, the green ball comes into contact with either a purple or blue object and remains in contact for at least 3 seconds.

Gravity points downward (−y), and all physical interactions follow Newtonian dynamics.

Notes:

Coordinates of objects are given by (x, y), normalized between 0 and 1,
where (0,0) is bottom-left and (1,1) is top-right of a 256×256 scene.

All objects fall under gravity except static ones (black and purple).
The dynamic flag in the object dict indicates if it can move.

During the simulation, objects move and collide according to Newtonian rules.

Action format:
You must return a single JSON array containing one or more “attempt” objects(suggested more tries).
Each object must have exactly these keys:

[
  {{
    "reason": "Why you chose this placement",
    "x": 0.123,   // normalized between 0 and 1
    "y": 0.456,   // normalized between 0 and 1
    "r": 0.03     // radius, e.g. 0.03
  }}
]
Only return the JSON array—no extra explanation, markdown, or text.
"""

        system_prompt = """You are a physics reasoning agent in a 2D environment governed by gravity."""
        try:
            messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
            ]
            generated_answer = call_llm(self.model,messages)
            print(generated_answer)
            return generated_answer
        except ValueError as e:
            return "json error"
        except Exception as e:
            return f"API error: {e}"

  
