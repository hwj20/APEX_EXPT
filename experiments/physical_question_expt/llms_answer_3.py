import os
import json
import copy
from experiments.physical_question_expt.utils.mujoco_perception import solve_problem
from experiments.physical_question_expt.utils.llm_router import call_llm  # 确保你已经实现这个函数

file_path = "./experiments/physical_question_expt/dataset/physics_questions.json"
with open(file_path, "r") as f:
    physics_questions = json.load(f)

import time  # ← 新增：计时功能

def run_model_on_questions(model, questions, result_path, with_perception=False, max_questions=3):
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    existing_question_texts = {r["question"]["question"] for r in existing_results}
    selected_questions = questions[:max_questions]

    for q in selected_questions:
        if q["question"] in existing_question_texts:
            print(f"Skipping already processed question: {q['question']}")
            continue

        ref = solve_problem(q, dt=0.001) if with_perception else None

        prompt = f"""
        Solve the following problem and return the answer in JSON format.

        Problem: {q["question"]}
        {"The external physical engine predictions: " + str(ref) if ref else ""}

        Expected JSON response:
        {{
            "reasoning": "Explanation of how you arrived at the answer"
            "answer": "Your final numerical answer(without unit and equation)" as {str(q['answer_json'])},
        }}

        Respond the JSON string only without any markdown symbol.
        """.strip()

        messages = [
            {"role": "system", "content": "You are a physics expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            start_time = time.time()  # ⏱ 开始时间
            response = call_llm(model, messages)
            end_time = time.time()    # ⏱ 结束时间
            duration = round(end_time - start_time, 3)

            print(f"⏱ Time taken: {duration}s")
            print(response)

            result_entry = {
                "question": q,
                "gpt4_response": response,
                "duration_seconds": duration  # ✅ 加入耗时记录
            }
            existing_results.append(result_entry)

        except Exception as e:
            end_time = time.time()
            duration = round(end_time - start_time, 3)

            print(f"❌ Error calling model {model}: {e} (in {duration}s)")
            existing_results.append({
                "question": q,
                "gpt4_response": {"error": str(e)},
                "duration_seconds": duration
            })

        with open(result_path, "w") as f:
            json.dump(existing_results, f, indent=4)

    return existing_results


if __name__ == "__main__":
    models = [
        # "gpt-4.1",
        # "tngtech/deepseek-r1t2-chimera:free",
        # "claude-sonnet-4-20250514",
        "gemini-2.5-flash",
        # "meta-llama/llama-4-scout",
    ]

    for model in models:
        ques = copy.deepcopy(physics_questions)

        # TEXT-ONLY
        result_path_plain = f"./experiments/physical_question_expt/results/{model.replace('/', '_').replace(':','_')}_physics_results_final.json"
        run_model_on_questions(model, ques, result_path_plain, with_perception=False, max_questions=200)

        # WITH PERCEPTION
        # ques = copy.deepcopy(physics_questions)
        # result_path_apex = f"./experiments/physical_question_expt/results/{model.replace('/', '_').replace(':','_')}_physics_results_final_APEX.json"
        # run_model_on_questions(model, ques, result_path_apex, with_perception=True, max_questions=200)


