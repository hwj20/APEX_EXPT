import json
import os

def evaluate_file(file_path):
    with open(file_path, 'r') as f:
        try:
            results = json.load(f)
        except Exception as e:
            print(f"[!] Failed to load {file_path}: {e}")
            return

    if not isinstance(results, list):
        print(f"[!] Skipping {file_path}: not a list of results.")
        return

    total = len(results)
    solved = sum(1 for r in results if r.get("solved") is True)
    avg_res_time = sum(r.get("res_time", 0) for r in results) / total if total else 0
    avg_sim_time = sum(r.get("sim_time", 0) for r in results) / total if total else 0
    avg_attemps = sum(len(json.loads(r.get("action", []))) for r in results) / total if total else 0

    print(f"📄 File: {os.path.basename(file_path)}")
    print(f"    Total tasks:      {total}")
    print(f"    Solved:           {solved} ({solved / total * 100:.2f}%)")
    print(f"    Avg response time: {avg_res_time:.3f}s")
    print(f"    Avg sim time:      {avg_sim_time:.3f}s")
    print(f"    ATT      {avg_attemps:.3f}")
    print("-" * 60)


def evaluate_all_json_files():
    current_dir = os.getcwd()
    files = [f for f in os.listdir(current_dir) if f.endswith(".json")]

    if not files:
        print("No JSON files found in current directory.")
        return

    for file in sorted(files):
        evaluate_file(os.path.join(current_dir, file))


if __name__ == "__main__":
    evaluate_all_json_files()
