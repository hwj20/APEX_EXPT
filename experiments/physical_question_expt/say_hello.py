import os
from llm_router import call_llm

models = [
    # "gpt-4",                   # OpenAI
    # "gpt-4o",                  # OpenAI
    "gpt-4.1",      # OpenAI 4.1
    "deepseek-chat",           # DeepSeek
    # "claude-3-opus-20240229",  # Claude 4
    # "gemini-2.0-flash",              # Gemini 2.5（Google）
    # "meta-llama/Llama-3-8b-chat-hf",  # HuggingFace LLaMA 3
]

prompt = "Say hello. Identify which model you are, and greet the user."

for model in models:
    print("=" * 50)
    print(f"🌟 Calling model: {model}")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        reply = call_llm(model, messages)
        print(f"✅ Response:\n{reply}")
    except Exception as e:
        print(f"❌ Error calling model {model}: {e}")
