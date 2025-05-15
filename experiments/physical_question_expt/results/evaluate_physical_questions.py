import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict

'''
This script is in a mess, horrible, evil, but correct.
'''

# Load JSON data
with open("../dataset/physics_ground_truth.json", "r") as f1:
    ground_truth_data = json.load(f1)

with open("gpt-4o_physics_results_final.json", "r") as f2:
    prediction_data = json.load(f2)

# Tolerance threshold
tolerance = 0.05

# parse json
for item in prediction_data:
    if "gpt4_response" in item:
        try:
            parsed = eval(item["gpt4_response"])
            if isinstance(parsed, dict) and "answer" in parsed:
                if isinstance(parsed['answer'], str):
                    item["answer_json"] = eval(parsed["answer"])
                else:
                    item["answer_json"] = parsed["answer"]
        except Exception as e:
            item["answer_json"] = {}

# Build mappings
gt_index = {item["question"]: item["answer_json"] for item in ground_truth_data}
pred_index = {item["question"]["question"]: item.get("answer_json", {}) for item in prediction_data}
gt_type_map = {item["question"]: item.get("type", "Unknown") for item in ground_truth_data}
para_index = {item["question"]: item["parameters"] for item in ground_truth_data}

# Evaluation stats
results = defaultdict(lambda: {"correct": 0, "total": 0, "mse_list": [], "numeric_valid": 0, "numeric_total": 0})


def safe_eval(expr):
    """
    Evaluates a mathematical expression.
    If there is an "=" in the expression, the result after "=" will be extracted.
    Otherwise, the expression will be evaluated directly.
    """
    try:
        if expr == '':
            return 0

        if "=" in expr:
            parts = expr.split("=")
            right_result = parts[-1].strip()
            try:
                return float(right_result)
            except ValueError:
                return None

        expr = expr.replace("^", "**")
        allowed_funcs = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "pi": math.pi, "e": math.e
        }
        return float(eval(expr, {"__builtins__": {}}, allowed_funcs))

    except:
        return None



# Compare predictions to ground truth
for question_text, ans1 in gt_index.items():
    ans2 = pred_index.get(question_text, {})
    task_type = gt_type_map[question_text]
    parameters = para_index[question_text]
    # delete the 'yz-plane'
    if task_type == "3D Circular Motion" and parameters.get("rotation_plane") == "yz-plane":
        continue
    if task_type == "3D Multi-Object Motion" and parameters.get("object_B", {}).get("rotation_plane") == "yz-plane":
        continue

    results[task_type]["total"] += 1
    if not ans2:
        continue

    matched = True
    total_sq_error = 0
    count = 0

    if task_type == '3D Collision' and ans1['will_collide'] == 'false' and ans2['will_collide'] == 'false':
        matched = True
    else:
        for key in ans1:
            if isinstance(ans1[key], dict) and isinstance(ans2.get(key), dict):
                for subkey in ans1[key]:
                    if subkey =='z_C':
                        continue
                    val = ans2[key].get(subkey)
                    if isinstance(val, str) and ("+" in val or "*" in val):
                        results[task_type]["numeric_total"] += 1
                    v1 = safe_eval(ans1[key][subkey]) if isinstance(ans1[key][subkey], str) else ans1[key][subkey]
                    v2 = safe_eval(ans2[key].get(subkey)) if isinstance(ans2[key].get(subkey), str) else ans2[key].get(
                        subkey)

                    if v1 is not None and v2 is not None and isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        total_sq_error += (v1 - v2) ** 2
                        count += 1
                        if abs(v1 - v2) > max(0.05, abs(tolerance * v1)):
                            matched = False
            else:
                if key == 'range_z':
                    continue
                v1 = safe_eval(ans1[key]) if isinstance(ans1[key], str) else ans1[key]
                v2 = safe_eval(ans2.get(key)) if isinstance(ans2.get(key), str) else ans2.get(key)

                val = ans2.get(key)
                if isinstance(val, str) and ("+" in val or "*" in val):
                    results[task_type]["numeric_total"] += 1

                if isinstance(v1, bool):
                    if v1 != v2:
                        matched = False
                elif v1 is not None and v2 is not None and isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    total_sq_error += (v1 - v2) ** 2
                    count += 1
                    if abs(v1 - v2) > max(0.05, abs(tolerance * v1)):
                        matched = False

    results[task_type]["mse_list"].append(0 if count == 0 else total_sq_error/count)

    if matched:
        results[task_type]["correct"] += 1
    else:
        print('-' * 20)
        print(ans1)
        print(ans2)

# Compile DataFrame
result_data = []
for task, stat in results.items():
    accuracy = stat["correct"] / stat["total"] * 100 if stat["total"] > 0 else 0
    mse = np.mean(stat["mse_list"]) if stat["mse_list"] else None
    numeric_rate = (stat["total"] - stat["numeric_total"]) / stat["total"] * 100 if stat["numeric_total"] >= 0 else None
    result_data.append({
        "Task Type": task,
        "Accuracy (%)": round(accuracy, 2),
        "MSE": round(mse, 4) if mse is not None else "N/A",
        "Numerical Validity (%)": round(numeric_rate, 2) if numeric_rate is not None else "N/A",
        "Total Samples": stat["total"]
    })

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df = pd.DataFrame(result_data)
print(df)
