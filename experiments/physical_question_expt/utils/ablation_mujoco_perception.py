import json
import time

import mujoco
import numpy as np
import pandas as pd
from mujoco_perception import *


# just to compare the sim results with ground truth (without LLM Answering)
def compare_answers_with_tolerance(file_name, tol=0.05):
    with open("../dataset/physics_ground_truth.json", "r") as f1:
        questions = json.load(f1)

    with open(file_name, "r") as f2:
        answers = json.load(f2)

    assert len(questions) == len(answers), "The number of questions are different"

    # Initialize statistics for each question type
    stats = {}
    for i in range(len(questions)):
        q = questions[i]
        ans_true = q.get("answer_json", {})
        ans_pred = answers[i].get("answer_json", {})

        q_type = q.get("type", "Unknown")
        if q_type not in stats:
            stats[q_type] = {"correct": 0, "total": 0}
        stats[q_type]["total"] += 1

        # Check structural mismatch
        if set(ans_true.keys()) != set(ans_pred.keys()):
            print(f"\n❗️[Mismatch Keys @ Q{i}] Type: {q_type}")
            print("Keys in ground truth:", ans_true.keys())
            print("Keys in prediction:", ans_pred.keys())
            continue

        # Compare values
        diff_found = False
        for key in ans_true:
            if isinstance(ans_true[key], dict) and isinstance(ans_pred[key], dict):
                for subkey in ans_true[key]:
                    try:
                        v1 = float(ans_true[key][subkey])
                        v2 = float(ans_pred[key][subkey])
                        if abs(v1 - v2) > max(0.05, abs(tol * v1)):
                            diff_found = True
                            break
                    except:
                        if ans_true[key][subkey] != ans_pred[key].get(subkey):
                            diff_found = True
                            break
            else:
                try:
                    v1 = float(ans_true[key])
                    v2 = float(ans_pred.get(key, ""))
                    if abs(v1 - v2) > max(0.05, abs(tol * v1)):
                        diff_found = True
                except:
                    if ans_true[key] != ans_pred.get(key):
                        diff_found = True
            if diff_found:
                break

        if not diff_found:
            stats[q_type]["correct"] += 1
        # else:
        # print(f"\n❗️[Mismatch Keys @ Q{i}]: {q_type}")
        # print("Keys in ground truth:", ans_true)
        # print("Keys in prediction:", ans_pred)

    # Print accuracy per question type
    print("\nAccuracy per question type:")
    for q_type, counts in stats.items():
        correct = counts["correct"]
        total = counts["total"]
        accuracy = (correct / total) * 100
        print(f" - {q_type}: {accuracy:.2f}% ({correct}/{total})")



if __name__ == "__main__":
    # Load the questions
    with open("../dataset/physics_questions.json", "r") as f:
        questions = json.load(f)

    dt_values = [0.001, 0.005, 0.01]

    for dt in dt_values:
        records = []
        results = []

        # Process each question
        for q in questions:
            start = time.time()
            answer = solve_problem(q, dt)
            elapsed = time.time() - start

            # Record timing
            records.append({
                "dt": dt,
                "question_type": q["type"],
                "duration_s": elapsed
            })

            # Save answer
            q_with_ans = q.copy()
            q_with_ans["answer_json"] = answer
            results.append(q_with_ans)

        # Save results for this dt
        out_path = f"../dataset/physics_answer_sim_dt_{dt}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for dt={dt} to {out_path}")

        # Compare answers with tolerance and print accuracy
        print(f"\nAccuracy for dt={dt}:")
        compare_answers_with_tolerance(out_path, tol=0.05)

        # Compute and display timing statistics
        df = pd.DataFrame(records)
        df_avg = df.groupby(["dt", "question_type"])["duration_s"].mean().reset_index()
        print(f"\nAverage duration per question type for dt={dt}:")
        print(df_avg[df_avg["dt"] == dt])
