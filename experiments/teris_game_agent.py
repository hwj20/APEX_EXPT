
import os

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
class LLM_Agent:
    def __init__(self, model="gpt-4"):
        self.model = model

    def decide_move(self, state):
        """调用 LLM 进行决策"""
        prompt = f"""
        You are an AI playing Tetris. Here is the current game board:
        {state}
        Decide the best move: 'left', 'right', 'rotate', or 'down'.
        
        return only one of the move without extra word or markdown symbol.
        """
        system_prompt = "You are a Tetris AI agent."
        try:
            client = OpenAI(
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
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


