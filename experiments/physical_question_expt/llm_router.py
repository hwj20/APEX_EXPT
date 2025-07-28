import os
from huggingface_hub import InferenceClient
import google.generativeai as genai
import openai
import anthropic
import requests
from openai import OpenAI as OpenAIClient
from anthropic import Anthropic

# 如果你之后还会加 DeepSeek / HuggingFace / 本地 LLaMA，也可以都加进来
#  OpenAI o3 o4 4.1 DeepSeek-R1, Claude 4,Gemini 2.5 Llama 3, 4

def call_llm(model, messages):
    """
    通用模型接口，输入统一的 messages（OpenAI 格式），自动调用正确的 LLM。
    """

    if model.startswith("gpt-"):
        client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5
        )
        return completion.choices[0].message.content

    elif model.startswith("claude-"):
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        completion = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=messages,
        )
        return completion.content[0].text

    elif model.startswith("deepseek-"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        api_base = "https://api.deepseek.com/v1"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.5
        }
        response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload)
        return response.json()
    elif model.startswith("gemini"):
        api_key = os.getenv("GEMINI_API_KEY")
        print(api_key)
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        # Gemini 使用 "contents" 而不是 "messages"，格式略不同
        user_content = {
            "contents": [
                {
                    "parts": [
                        {"text": messages[-1]["content"]}
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key
        }

        response = requests.post(api_url, headers=headers, json=user_content)
        data = response.json()

        # 取出生成内容
        return data["candidates"][0]["content"]["parts"][0]["text"]
    elif "llama" in model:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        client = InferenceClient(model=model, token=hf_token)
        prompt_text = messages[-1]["content"]
        return client.text_generation(prompt_text, max_new_tokens=200)


    else:
        raise ValueError(f"Unknown model: {model}")

