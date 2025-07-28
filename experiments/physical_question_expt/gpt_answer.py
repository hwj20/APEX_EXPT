import copy
import json
import os

from openai import OpenAI
from experiments.physical_question_expt.utils.mujoco_perception import solve_problem

# question path
file_path = "./experiments/physical_question_expt/dataset/physics_questions.json"
with open(file_path, "r") as f:
    physics_questions = json.load(f)

api_key = os.getenv("OPENAI_API_KEY")


def ask_gpt4_with_perception(result_path, questions, model, max_questions=200):
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

    # find solved questions
    existing_question_texts = {r["question"]["question"] for r in existing_results}

    selected_questions = questions[:max_questions]

    for q in selected_questions:
        if q["question"] in existing_question_texts:
            with open(result_path, "w") as f:
                json.dump(existing_results, f, indent=4)
            print(f"Skipping already processed question: {q['question']}")
            continue

        # Note: this step will change "answer_json" in q; but we will not use it as the answer in the end
        ref = solve_problem(q,dt=0.001)

        prompt = f"""
        Solve the following problem and return the answer in JSON format.

        Problem: {q["question"]}
        The external physical engine predictions: {ref}

        Expected JSON response:
        {{
            "reasoning": "Explanation of how you arrived at the answer"
            "answer": "Your final numerical answer(without unit and equation)" as {str(q['answer_json'])},
        }}

        Respond the JSON string only without any markdown symbol.
        """

        try:
            client = OpenAI(
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a physics expert."},
                    {"role": "user", "content": prompt}],
                temperature=0.5
            )
            gpt_answer = str(completion.choices[0].message.content)
            print(gpt_answer)
            # append answer
            result_entry = {"question": q, "gpt4_response": gpt_answer}
            existing_results.append(result_entry)

            with open(result_path, "w") as f:
                json.dump(existing_results, f, indent=4)

        except Exception as e:
            existing_results.append({"question": q, "gpt4_response": {"error": str(e)}})

    return existing_results


def ask_gpt4(result_path, questions, model, max_questions=200):
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

    existing_question_texts = {r["question"]["question"] for r in existing_results}

    selected_questions = questions[:max_questions]

    for q in selected_questions:
        if q["question"] in existing_question_texts:
            with open(result_path, "w") as f:
                json.dump(existing_results, f, indent=4)
            print(f"Skipping already processed question: {q['question']}")
            continue  # Skip the questions solved
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

        try:
            client = OpenAI(
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a physics expert."},
                    {"role": "user", "content": prompt}],
                temperature=0.5
            )
            gpt_answer = str(completion.choices[0].message.content)
            print(gpt_answer)
            # Append result
            result_entry = {"question": q, "gpt4_response": gpt_answer}
            existing_results.append(result_entry)

            # Save to file for each question
            with open(result_path, "w") as f:
                json.dump(existing_results, f, indent=4)

        except Exception as e:
            existing_results.append({"question": q, "gpt4_response": {"error": str(e)}})

    return existing_results


# This expt will run for about one hour
# my suggestion is to run it in multi scripts, like one script for one model

models = ['gpt-4o', 'gpt-4o-mini']
for model in models:
    ques = copy.deepcopy(physics_questions)
    gpt4_results_path = f"results/{model}_physics_results_final.json"
    gpt4_results = ask_gpt4(gpt4_results_path, ques, model, max_questions=200)

    ques = copy.deepcopy(physics_questions)
    gpt4_results_path = f"results/{model}_physics_results_final_APEX.json"
    gpt4_results = ask_gpt4_with_perception(gpt4_results_path, ques, model, max_questions=200)

# The following code is designed as if you wish to rerun a certain kind of questions, not all

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
#         continue  # skip the question solved before
#
# print(len(results))
# with open(gpt4_results_path, "w") as f:
#     json.dump(results, f, indent=4)
