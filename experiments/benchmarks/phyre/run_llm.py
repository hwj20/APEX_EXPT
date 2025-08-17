from PhyreAgent import *
import sys

agent = LLM_Agent(model='gpt-4.1')

def main():
    if len(sys.argv) != 3:
        print("Usage: python router_bridge.py <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    # Load input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Call the router function
    # print(data)
    result = agent.decide_move(data)

    # Write output JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

main()