import json
import os

from openai import OpenAI

# 读取物理题
file_path = "dataset/physics_questions.json"
with open(file_path, "r") as f:
    physics_questions = json.load(f)

api_key = os.getenv("OPENAI_API_KEY")


# GPT-4 API 调用函数（模拟批量考试）
def ask_gpt4(questions, max_questions=5):
    results = []

    # 限制测试题目数量，避免消耗过多计算资源
    selected_questions = questions[:max_questions]

    for q in selected_questions:
        prompt = f"""
        Solve the following problem and return the answer in JSON format.

        Problem: {q["question"]}

        Expected JSON response:
        {{
            "reasoning": "Explanation of how you arrived at the answer"
            "answer": "Your final numerical answer(without unit)",
        }}
        
        Respond the JSON string only without any markdown symbol.
        """

        # 调用 GPT-4 API 进行物理题解答
        try:
            client = OpenAI(
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a physics expert."},
                    {"role": "user", "content": prompt}],
                temperature=0.5
            )
            gpt_answer = str(completion.choices[0].message.content)
            print(gpt_answer)
            results.append({"question": q, "gpt4_response": gpt_answer})
        except Exception as e:
            results.append({"question": q, "gpt4_response": {"error": str(e)}})

    return results


# 运行 GPT-4 物理考试（测试 5 题）
gpt4_results = ask_gpt4(physics_questions, max_questions=5)

# 保存 GPT-4 答案
gpt4_results_path = "gpt4_physics_results.json"
with open(gpt4_results_path, "w") as f:
    json.dump(gpt4_results, f, indent=4)
