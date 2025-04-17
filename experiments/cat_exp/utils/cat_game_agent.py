import json
import os
import re

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
def strip_markdown(text: str) -> str:
    # 去除代码块标记 ```json ```python等
    text = re.sub(r"```", "", text)
    text = re.sub("json", "", text)
    # 去除标题 #
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # 去除加粗/斜体 ** ** 或 __ __ 或 * *
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    # 去除行内代码 `
    text = re.sub(r"`(.*?)`", r"\1", text)
    # 去除列表项 - *
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    # 去除多余空行
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# Control Policy
def decode(decision: str):
    """
    输入 LLM 决策的字符串，返回 {'velocity': [...], 'duration': ...}
    """
    decision = decision.lower()  # 防止大小写影响
    decision = json.loads(decision)

    # stay
    vel = [0.0, 0.0, 0.0]  # x, y, z
    duration = decision['duration']  # s

    if "left" in decision['move']:
        vel[0] = -3.0
    elif "right" in decision['move']:
        vel[0] = 3.0
    elif "up" in decision['move']:
        vel[1] = 3.0
    elif "down" in decision['move']:
        vel[1] = -3.0
    elif "jump" in decision['move']:
        vel[2] = 3.0  # jump
        duration = 0.1

    return {"velocity": vel, "duration": duration}

class LLM_Agent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=api_key)


    def decide_move_apex(self,state,summary, available_move, apex_results):
        prompt = f"""
        You are controlling a robot in a 3D physical environment with moving obstacles.
        Your goal is to avoid collisions with cats while progressing toward the target location.

        Current state:
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
            print(generated_answer)
            return decode(strip_markdown(generated_answer))
        except Exception as e:
            return f"API error: {e}"

    def decide_move(self, state, available_move):
        prompt = f"""
        You are controlling a robot in a 3D physical environment with moving obstacles.
        Your goal is to avoid collisions with cats while progressing toward the target location.

        Current state:
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
            print(generated_answer)
            return decode(strip_markdown(generated_answer))
        except Exception as e:
            return f"API error: {e}"

