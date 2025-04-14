import json
import os

from openai import OpenAI
from experiments.utils.mujoco_perception import solve_problem

# 读取物理题
file_path = "dataset/physics_questions.json"
with open(file_path, "r") as f:
    physics_questions = json.load(f)

api_key = os.getenv("OPENAI_API_KEY")


def ask_gpt4_with_perception(result_path, questions, max_questions=200):
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            try:
                existing_results = json.load(f)
                # if you want to recalculate a type of problem
                # existing_results = [q for q in existing_results if q["question"]["type"] != "3D Collision"]
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    # 计算已完成的题目数量
    existing_question_texts = {r["question"]["question"] for r in existing_results}
    # results = []

    # 限制测试题目数量，避免消耗过多计算资源
    selected_questions = questions[:max_questions]

    for q in selected_questions:
        if q["question"] in existing_question_texts:
            with open(result_path, "w") as f:
                json.dump(existing_results, f, indent=4)
            print(f"Skipping already processed question: {q['question']}")
            continue

        # this step will change "answer_json" in q but we will only count the answer_json responded by LLM
        ref = solve_problem(q)

        prompt = f"""
        Solve the following problem and return the answer in JSON format.

        Problem: {q["question"]}
        The external physical engine calculations: {ref}

        Expected JSON response:
        {{
            "reasoning": "Explanation of how you arrived at the answer"
            "answer": "Your final numerical answer(without unit and equation)" as {str(q['answer_json'])},
        }}

        Respond the JSON string only without any markdown symbol.
        """

        # 调用 GPT-4 API 进行物理题解答
        try:
            client = OpenAI(
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a physics expert."},
                    {"role": "user", "content": prompt}],
                temperature=0.5
            )
            gpt_answer = str(completion.choices[0].message.content)
            print(gpt_answer)
            # 追加结果
            result_entry = {"question": q, "gpt4_response": gpt_answer}
            existing_results.append(result_entry)

            # 立即保存到文件，防止 API 崩溃导致数据丢失
            with open(result_path, "w") as f:
                json.dump(existing_results, f, indent=4)

        except Exception as e:
            existing_results.append({"question": q, "gpt4_response": {"error": str(e)}})

    return existing_results


# GPT-4 API 调用函数（模拟批量考试）
def ask_gpt4(result_path, questions, max_questions=200):
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            try:
                existing_results = json.load(f)
                # if you want to recalculate a type of problem
                # existing_results = [q for q in existing_results if q["question"]["type"] != "3D Collision"]
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    # 计算已完成的题目数量
    existing_question_texts = {r["question"]["question"] for r in existing_results}
    # results = []

    # 限制测试题目数量，避免消耗过多计算资源
    selected_questions = questions[:max_questions]

    for q in selected_questions:
        if q["question"] in existing_question_texts:
            with open(result_path, "w") as f:
                json.dump(existing_results, f, indent=4)
            print(f"Skipping already processed question: {q['question']}")
            continue  # 如果已经解过这道题，跳过
        prompt = f"""
        Solve the following problem and return the answer in JSON format.

        Problem: {q["question"]}

        Expected JSON response:
        {{
            "reasoning": "Explanation of how you arrived at the answer"
            "answer": "Your final numerical answer(without unit and equation)" as {str(q['answer_json'])},
        }}
        
        Respond the JSON string only without any markdown symbol.
        """

        # 调用 GPT-4 API 进行物理题解答
        try:
            client = OpenAI(
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a physics expert."},
                    {"role": "user", "content": prompt}],
                temperature=0.5
            )
            gpt_answer = str(completion.choices[0].message.content)
            print(gpt_answer)
            # 追加结果
            result_entry = {"question": q, "gpt4_response": gpt_answer}
            existing_results.append(result_entry)

            # 立即保存到文件，防止 API 崩溃导致数据丢失
            with open(result_path, "w") as f:
                json.dump(existing_results, f, indent=4)

        except Exception as e:
            existing_results.append({"question": q, "gpt4_response": {"error": str(e)}})

    return existing_results


# gpt4_results_path = "gpt4_physics_results_final.json"
# gpt4_results = ask_gpt4(gpt4_results_path, physics_questions, max_questions=200)
gpt4_results_path = "results/gpt4_mini_physics_results_final_APEX.json"
gpt4_results = ask_gpt4_with_perception(gpt4_results_path, physics_questions, max_questions=200)

# if os.path.exists(gpt4_results_path):
#     with open(gpt4_results_path, "r") as f:
#         try:
#             existing_results = json.load(f)
#             existing_results = existing_results[25:]
#         except json.JSONDecodeError:
#             existing_results = []
# else:
#     existing_results = []
# with open("gpt4_physics_results.json") as f:
#     to_merge = json.load(f)
#     to_merge = [q for q in to_merge if q["question"]["type"] != "3D Circular Motion"]
# existing_question_texts = {r["question"]["question"] for r in existing_results}
# results = existing_results
# for q in to_merge:
#     if q["question"]["question"] not in existing_question_texts:
#         results.append(q)
#         continue  # 如果已经解过这道题，跳过
#
# print(len(results))
# with open(gpt4_results_path, "w") as f:
#     json.dump(results, f, indent=4)
