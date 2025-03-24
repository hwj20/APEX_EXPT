import base64
import requests
import os

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")


class LLM_Agent:
    def __init__(self, model="gpt-4o"):
        self.model = model

    # PGD
    def decide_move_pgd(self, state, pgd_results):
        prompt = f"""
        You are playing Tetris. Your goal is to maximize the score by:
        - Clearing as many lines as possible.
        - Keeping the board as flat as possible.
        - Avoiding unnecessary stacking.

        Here is the current board state(0-blank,1-landed piece,2-current piece):
        {state}
        
        Here are physical engine analysis:{pgd_results}

        Available moves:
        - "left": Move the piece left by one column.
        - "right": Move the piece right by one column.
        - "rotate": Rotate the piece 90 degrees clockwise.
        - "down": Instantly drop the piece to the lowest possible position.(max times = 1)
        
        Decide the best move sequence in JSON format as a list of actions. Each action should include the move and how many times to perform it.

        Example:
        [
          {{"move": "left", "times": 2}},
          {{"move": "rotate", "times": 1}},
          {{"move": "down", "times": 1}}
        ]

        Allowed moves are: "left", "right", "rotate", and "down".
        Only return the JSON array without any explanation or markdown. No Markdown
        """

        system_prompt = "You are a Tetris AI agent."
        try:
            client = OpenAI(
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            generated_answer = str(completion.choices[0].message.content)
            return generated_answer
        except Exception as e:
            return f"API error: {e}"

    # baseline
    def decide_move(self, state,model):
        """调用 LLM 进行决策"""
        prompt = f"""
        You are playing Tetris. Your goal is to maximize the score by:
        - Clearing as many lines as possible.
        - Keeping the board as flat as possible.
        - Avoiding unnecessary stacking.

        Here is the current board state(0-blank,1-landed piece,2-current piece):
        {state}

        Available moves:
        - "left": Move the piece left by one column.
        - "right": Move the piece right by one column.
        - "rotate": Rotate the piece 90 degrees clockwise.
        - "down": Instantly drop the piece to the lowest possible position.(max times = 1)
        
        Decide the best move sequence in JSON format as a list of actions. Each action should include the move and how many times to perform it.

        Example:
        [
          {{"move": "left", "times": 2}},
          {{"move": "rotate", "times": 1}},
          {{"move": "down", "times": 1}}
        ]

        Allowed moves are: "left", "right", "rotate", and "down".
        Only return the JSON array without any explanation or markdown. No Markdown
        """

        system_prompt = "You are a Tetris AI agent."
        try:
            client = OpenAI(
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            generated_answer = str(completion.choices[0].message.content)
            return generated_answer
        except Exception as e:
            return f"API error: {e}"
    def vlm_decide_move_pgd(self, image_path,pgd_results):
        prompt = f"""
        You are playing Tetris. Your goal is to maximize the score by:
        - Clearing as many lines as possible.
        - Keeping the board as flat as possible.
        - Avoiding unnecessary stacking.
        
        Here are physical engine analysis:{pgd_results}

        Available moves:
        - "left": Move the piece left by one column.
        - "right": Move the piece right by one column.
        - "rotate": Rotate the piece 90 degrees clockwise.
        - "down": Instantly drop the piece to the lowest possible position.(max times = 1)

        Decide the best move sequence in JSON format as a list of actions. Each action should include the move and how many times to perform it.

        Example:
        [
          {{"move": "left", "times": 2}},
          {{"move": "rotate", "times": 1}},
          {{"move": "down", "times": 1}}
        ]

        Allowed moves are: "left", "right", "rotate", and "down". 
        
        Only return the JSON array without any explanation or markdown. No Markdown

        Here is the current board state:
        """

        system_prompt = "You are a Tetris AI agent."

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

            message = response_data["choices"][0]["message"]["content"]
            return message
        except Exception as e:
            return f"API error: {e}"

    def vlm_decide_move(self, image_path):
        prompt = f"""
        You are playing Tetris. Your goal is to maximize the score by:
        - Clearing as many lines as possible.
        - Keeping the board as flat as possible.
        - Avoiding unnecessary stacking.

        Available moves:
        - "left": Move the piece left by one column.
        - "right": Move the piece right by one column.
        - "rotate": Rotate the piece 90 degrees clockwise.
        - "down": Instantly drop the piece to the lowest possible position.(max times = 1)

        Example:
        [
          {{"move": "left", "times": 2}},
          {{"move": "rotate", "times": 1}},
          {{"move": "down", "times": 1}}
        ]

        Allowed moves are: "left", "right", "rotate", and "down".
        Only return the JSON array without any explanation or markdown. No Markdown

        Here is the current board state:
        """

        system_prompt = "You are a Tetris AI agent."

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

            message = response_data["choices"][0]["message"]["content"]
            return message
        except Exception as e:
            return f"API error: {e}"
