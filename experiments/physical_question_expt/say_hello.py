import os
from experiments.physical_question_expt.utils.llm_router import call_llm
# TODO 新的实验
models = [
    "gpt-4.1",      # OpenAI 4.1
    "tngtech/deepseek-r1t2-chimera:free",           # DeepSeek r1
    "claude-sonnet-4-20250514",  # Claude 4
    "gemini-2.5-flash",              # Gemini 2.5（Google）
    "meta-llama/llama-4-scout",  # HuggingFace LLaMA 4
]

prompt = "Say hello. Identify which model you are, and greet the user."
system_prompt =  "You are a physics expert."

for model in models:
    print("=" * 50)
    print(f"🌟 Calling model: {model}")
    try:
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
        ]
        reply = call_llm(model, messages)
        print(f"✅ Response:\n{reply}")
    except Exception as e:
        print(f"❌ Error calling model {model}: {e}")
