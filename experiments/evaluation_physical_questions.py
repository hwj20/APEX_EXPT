# Re-import necessary modules and rerun evaluation after code state reset
import json
import math

import numpy as np
from collections import defaultdict
import pandas as pd

# Re-uploaded file paths
ground_truth_path = "../dataset/physics_ground_truth.json"
predicted_path = "../dataset/gpt4_physics_results_final_APEX.json"

# Tolerance threshold
tolerance = 0.05

# Load JSON data
with open(ground_truth_path, "r") as f1:
    ground_truth_data = json.load(f1)

with open(predicted_path, "r") as f2:
    prediction_data = json.load(f2)

# 预处理：将字符串形式的 JSON 解析成 dict，并提取出其中的 "answer"
for item in prediction_data:
    if "gpt4_response" in item:
        try:
            parsed = eval(item["gpt4_response"])  # 危险区域开启，宝子请小心（笑）
            if isinstance(parsed, dict) and "answer" in parsed:
                if isinstance(parsed['answer'], str):
                    # 再 eval 一次嵌套 answer 字符串
                    item["answer_json"] = eval(parsed["answer"])
                else:
                    item["answer_json"] = parsed["answer"]
        except Exception as e:
            print(f"❗️解析失败：第 {prediction_data.index(item)} 题 gpt4_response 非法 Python 表达式")
            print("错误内容：", e)
            print("内容如下：\n", item['gpt4_response'])
    else:
        print("No response found for item index", prediction_data.index(item))

# Build mapping from question text to answer_json
gt_index = {item["question"]: item["answer_json"] for item in ground_truth_data}
pred_index = {item["question"]["question"]: item["answer_json"] for item in prediction_data}
gt_type_map = {item["question"]: item.get("type", "Unknown") for item in ground_truth_data}

# Evaluation stats
results = defaultdict(lambda: {"correct": 0, "total": 0})


def safe_eval(expr):
    try:
        if expr == '':
            return 0
        expr = expr.replace("^", "**")  # 替换数学表达式中的 ^

        if expr.lower() == 'true' or expr.lower() == 'false':
            return bool(expr)
        else:
            allowed_funcs = {
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "pi": math.pi,
                "e": math.e
            }
            return float(eval(expr, {"__builtins__": {}}, allowed_funcs))

    except Exception as e:
        print(f"❗️表达式计算失败：{expr}")
        return None


# Iterate over ground truth questions
for question_text, ans1 in gt_index.items():
    ans2 = pred_index.get(question_text, {})
    task_type = gt_type_map[question_text]
    results[task_type]["total"] += 1
    if not ans2:
        continue

    matched = True
    if set(ans1.keys()) != set(ans2.keys()):
        matched = False
    else:
        for key in ans1:
            try:
                # this is a fault in ground truth, then delete comparsion on  this variable
                if key == 'range_z':
                    continue
                if isinstance(ans1[key], dict) and isinstance(ans2.get(key), dict):
                    for subkey in ans1[key]:
                        if isinstance(ans1[key][subkey], str):
                            v1 = safe_eval(ans1[key][subkey])
                        else:
                            v1 = ans1[key][subkey]
                        if isinstance(ans2[key][subkey], str):
                            v2 = safe_eval(ans2[key][subkey])
                        else:
                            v2 = ans2[key][subkey]
                        if isinstance(v1, bool):
                            if v1 != v2:
                                matched = False
                                break
                            continue
                        if abs(v1 - v2) > abs(tolerance * v1):
                            matched = False
                            break
                else:
                    if isinstance(ans1[key], str):
                        v1 = safe_eval(ans1[key])
                    else:
                        v1 = ans1[key]

                    if isinstance(ans2[key], str):
                        v2 = safe_eval(ans2[key])
                    else:
                        v2 = ans2[key]
                    if isinstance(v1, bool):
                        if v1 != v2:
                            matched = False
                            break
                        continue
                    if abs(v1 - v2) > abs(tolerance * v1):
                        matched = False
                        break
            except:
                if ans1[key] != ans2.get(key):
                    print(ans1)
                    print(ans2)
                    matched = False
                    break

    if matched:
        results[task_type]["correct"] += 1

# Compile results into DataFrame
result_data = []
for task, stat in results.items():
    accuracy = stat["correct"] / stat["total"] * 100 if stat["total"] > 0 else 0
    result_data.append({
        "Task Type": task,
        "Accuracy (%)": round(accuracy, 2),
        "Total Samples": stat["total"]
    })

df = pd.DataFrame(result_data)
print(df)
