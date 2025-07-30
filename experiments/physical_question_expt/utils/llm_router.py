import os
from openai import OpenAI as OpenRouterClient
from huggingface_hub import InferenceClient
import google.generativeai as genai
import openai
import anthropic
import requests
from openai import OpenAI as OpenAIClient
from anthropic import Anthropic

# def call_llm_with_vision(model, messages, img_url):
#     """
#     支持图像输入的模型接口。
#     要求 messages 是 OpenAI chat 格式，img_url 是图像的公网链接或 base64 URL。
#     """
#     # 插入 vision 模态内容到最后一条 user message 中
#     if messages[-1]["role"] != "user":
#         raise ValueError("最后一条 message 必须是 user 角色才能附加图片")

#     text = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else ""
#     messages[-1]["content"] = [
#         {
#             "type": "text",
#             "text": text
#         },
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": img_url
#             }
#         }
#     ]

#     # GPT-4 Vision
#     if model.startswith("gpt-"):
#         client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
#         completion = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=0.5
#         )
#         return completion.choices[0].message.content

#     # Claude Vision
#     elif model.startswith("claude-"):
#         client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

#         system_prompt = ""
#         real_messages = []
#         for m in messages:
#             if m["role"] == "system":
#                 system_prompt = m["content"]
#             else:
#                 real_messages.append(m)

#         completion = client.messages.create(
#             model=model,
#             max_tokens=1000,
#             system=system_prompt,
#             messages=real_messages
#         )
#         return completion.content[0].text

#     # Gemini Vision
#     elif model.startswith("gemini"):
#         api_key = os.getenv("GEMINI_API_KEY")
#         api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#         headers = {
#             "Content-Type": "application/json",
#             "X-goog-api-key": api_key
#         }

#         # Gemini 接收 "contents"，嵌套结构
#         user_content = {
#             "contents": [
#                 {
#                     "role": "user",
#                     "parts": messages[-1]["content"]
#                 }
#             ]
#         }

#         response = requests.post(api_url, headers=headers, json=user_content)
#         data = response.json()
#         return data["candidates"][0]["content"]["parts"][0]["text"]

#     # LLaMA-4 Vision via OpenRouter
#     elif model.startswith("meta-llama/") or "openrouter" in model:
#         from openai import OpenAI
#         client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=os.getenv("OPENROUTER_API_KEY"),
#         )

#         completion = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=0.5,
#             extra_headers={
#                 "HTTP-Referer": "https://apex.com",
#                 "X-Title": "ApexAgent",
#             },
#             extra_body={}
#         )
#         return completion.choices[0].message.content

#     else:
#         raise ValueError(f"Model `{model}` is unknown or does not support vision.")


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

        # 提取 system message（如果有）
        system_prompt = ""
        real_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                real_messages.append(m)

        completion = client.messages.create(
            model=model,
            max_tokens=1000,
            system=system_prompt,
            messages=real_messages
        )
        return completion.content[0].text

    elif model.startswith("deepseek/") or "openrouter.ai" in os.getenv("OPENROUTER_API_KEY", ""):

        client = OpenRouterClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            extra_headers={
                "HTTP-Referer": "https://apex.com",  # 可自定义
                "X-Title": "ApexAgent",  # 可自定义
            },
            extra_body={}
        )

        return completion.choices[0].message.content

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
    elif "openrouter" in model or model.startswith("meta-llama/"):
        client = OpenRouterClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            extra_headers={
                "HTTP-Referer": "https://apex.com",  # 可自定义
                "X-Title": "ApexAgent",  # 可自定义
            },
            extra_body={}
        )

        return completion.choices[0].message.content


    else:
        raise ValueError(f"Unknown model: {model}")

