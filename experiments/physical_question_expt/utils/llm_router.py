import mimetypes
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import base64
import os
from openai import OpenAI as OpenAIClient
from anthropic import Anthropic
from openai import OpenAI as OpenRouterClient
import requests


def call_llm(model, messages, img_url=None):
    """
    通用模型接口，支持可选的图像输入。
    - model: 模型标识
    - messages: OpenAI-style chat 消息列表
    - img_url: 可选，图像的公网链接或 base64 data-url
    """
    # 如果提供了 img_url，将其注入到最后一条 user message
    if img_url:
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("最后一条 message 必须是 user 角色才能附加图片")
        # 提取纯文本
        original = messages[-1]["content"]
        if isinstance(original, list):
            text = " ".join([p.get("text", "") for p in original if p.get("type") == "text"]).strip()
        else:
            text = original
        # 处理本地路径或 URL
        if img_url.startswith("http://") or img_url.startswith("https://") or img_url.startswith("data:"):
            data_url = img_url
        else:
            # 视为本地文件路径，读取并转 base64
            mime, _ = mimetypes.guess_type(img_url)
            with open(img_url, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:{mime or 'application/octet-stream'};base64,{b64}"
        # 构造混合消息
        messages[-1]["content"] = [
            {"type": "text",      "text": text},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]


    # GPT 系列（支持 Vision Chat）
    if model.startswith("gpt-"):
        client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5
        )
        return completion.choices[0].message.content

    # Claude 多模态，使用原生 image/message schema
    elif model.startswith('claude-'):
        client = Anthropic()
        real_msgs = []
        for m in messages:
            if m['role'] == 'system':
                # Claude SDK 不支持单独 system 字段，作为首条 user 消息注入
                real_msgs.append({'role':'user','content':[{'type':'text','text':m['content']}]})
            else:
                content_list = []
                if isinstance(m['content'], list):
                    for part in m['content']:
                        if part['type'] == 'text':
                            content_list.append({'type':'text','text': part['text']})
                        elif part['type'] == 'image_url':
                            url = part['image_url']['url']
                            if url.startswith('data:'):
                                header, b64 = url.split(',',1)
                                media_type = header.split(';')[0].split(':')[1]
                                content_list.append({
                                    'type':'image',
                                    'source':{
                                        'type':'base64',
                                        'media_type': media_type,
                                        'data': b64
                                    }
                                })
                            else:
                                content_list.append({
                                    'type':'image',
                                    'source':{
                                        'type':'url',
                                        'uri': url
                                    }
                                })
                else:
                    content_list.append({'type':'text','text': m['content']})
                real_msgs.append({'role':'user','content': content_list})
        completion = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=real_msgs
        )
        return completion.content[0].text

    elif model.startswith('gemini'):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_obj = genai.GenerativeModel(model)

            # Get the content of the last message
            last_message_content = messages[-1]['content']
            parts = []

            if isinstance(last_message_content, list):
                # Handle multipart content (text and images)
                for part in last_message_content:
                    if part.get('type') == 'text':
                        parts.append(part['text'])
                    elif part.get('type') == 'image_url':
                        image_url = part['image_url']['url']
                        if image_url.startswith("data:"):
                            # Handle data URI
                            header, encoded_data = image_url.split(",", 1)
                            image_bytes = base64.b64decode(encoded_data)
                            img = Image.open(BytesIO(image_bytes))
                            parts.append(img)
                        else:
                            # Handle regular URL
                            response = requests.get(image_url)
                            response.raise_for_status()  # Raise an exception for bad status codes
                            img = Image.open(BytesIO(response.content))
                            parts.append(img)
            else:
                # Handle simple text content
                parts.append(last_message_content)

            # Generate content with the processed parts
            response = model_obj.generate_content(contents=parts)
            return response.text

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Error fetching or processing image: {e}")
            return "Error: Failed to process image for Gemini request."
        except Exception as e:
            print(f"Gemini SDK request failed: {e}")
            return "Error: Gemini SDK request failed."

    # OpenRouter / LLaMA-4 Vision
    elif model.startswith("meta-llama/") or model.startswith("deepseek/") or "openrouter" in model:
        client = OpenRouterClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5
        )
        return completion.choices[0].message.content

    else:
        raise ValueError(f"Unknown model or no vision support: {model}")

# 示例调用：
# resp = call_llm('gpt-4o', messages, img_url='https://cdn.example.com/scene.jpg')
# print(resp)
