import json
from collections import defaultdict

# 用于统计每种类型的总时间和数量
type_count = defaultdict(int)
models = [
    "gpt-4.1",      # OpenAI 4.1
    "deepseek/deepseek-r1-0528",           # DeepSeek r1
    "claude-sonnet-4-20250514",  # Claude 4
    "gemini-2.5-flash",              # Gemini 2.5（Google）
    "meta-llama/llama-4-scout",  # HuggingFace LLaMA 4
]


for model in models:
    file_path = f"{model.replace('/', '_')}_physics_results_final.json"
    print(model)
    type_count = defaultdict(int)
    type_time = defaultdict(float)

    with open(file_path, "r") as f2:
        prediction_data = json.load(f2)
        for item in prediction_data:
            task_type = item['question']['type']
            duration = item['duration_seconds']
            type_time[task_type] += duration
            type_count[task_type] += 1

    # 输出每种任务类型的平均时长
    for task_type in type_time:
        avg_time = type_time[task_type] / type_count[task_type]
        print(f"{task_type}: {avg_time:.3f} seconds (over {type_count[task_type]} samples)")
