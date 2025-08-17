import time
from experiments.tetris_expt.utils.tetris_game_agent_other_llms import *
from experiments.tetris_expt.utils.Tetris import *

results = {
    "gpt-4.1": {
        "final_score": [],
        "max_stack_height": [],
        "holes": [],
        "bumps": [],
        "height_delta_per_move": [],
        "resp_time": []
    },
    "claude-sonnet-4-20250514": {
        "final_score": [],
        "max_stack_height": [],
        "holes": [],
        "bumps": [],
        "height_delta_per_move": [],
        "resp_time": []
    },
    "gemini-2.5-flash": {
        "final_score": [],
        "max_stack_height": [],
        "holes": [],
        "bumps": [],
        "height_delta_per_move": [],
        "resp_time": []
    },
    "meta-llama/llama-4-scout": {
        "final_score": [],
        "max_stack_height": [],
        "holes": [],
        "bumps": [],
        "height_delta_per_move": [],
        "resp_time": []
    }
}


def run_tetris(method, save_path="demo_", save_type=".json", rng=random.Random(42), auto_gen_path=False):
    pygame.init()
    MAX_EPOCH = 30  # 1 piece/2 epochs
    if auto_gen_path:
        save_path = save_path + method + str(MAX_EPOCH) + save_type
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    tetris = Tetris(rng=rng)

    # initialize agents
    ag = LLM_Agent(model=method)

    # game history
    game_history = []
    resp_times = []
    cnt = 0

    while tetris.running:
        if cnt < MAX_EPOCH:
            cnt += 1
        else:
            break
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                tetris.running = False

        state = tetris.get_state()

        if state != tetris.previous_state and tetris.piece_y > 1:
            tetris.render(screen)
            action_json = ''
            # measure response time for each decision
            start_time = time.time()
            if method == 'APEX':
                APEX_results = tetris.apex_evaluate()
                action_json = ag.decide_move_APEX(state, APEX_results)
            else:
                action_json = ag.decide_move(state, method)

            if action_json is None:
                action_json = '{"move":"down","times":1}'
            
            end_time = time.time()

            resp_time = end_time - start_time
            resp_times.append(resp_time)

            game_history.append({
                "board": state["board"],
                "piece": state["piece"],
                "piece_x": state["piece_x"],
                "piece_y": state["piece_y"],
                "action": action_json,
                "score": state["score"],
                "resp_time": resp_time
            })

            tetris.previous_state = state
            tetris.step(action_json)

        tetris.gravity()
        tetris.render(screen)
        clock.tick(1)

    pygame.quit()

    with open(save_path, "w") as f:
        json.dump(game_history, f, indent=4)
    print(f"game history is saved: {save_path}")

    # aggregate evaluation metrics and average response time
    evals = tetris.final_evaluation()
    avg_resp_time = sum(resp_times) / len(resp_times) if resp_times else 0
    evals["avg_resp_time"] = avg_resp_time
    return evals


def if_you_want_to_play_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    tetris = Tetris()

    while tetris.running:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                tetris.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    tetris.step("{\"move\": \"left\", \"times\": 1}")
                elif event.key == pygame.K_RIGHT:
                    tetris.step("{\"move\": \"right\", \"times\": 1}")
                elif event.key == pygame.K_UP:
                    tetris.step("{\"move\": \"rotate\", \"times\": 1}")
                elif event.key == pygame.K_DOWN:
                    tetris.step("{\"move\": \"down\", \"times\": 1}")

        tetris.gravity()
        tetris.render(screen)
        clock.tick(1)

    pygame.quit()


if __name__ == "__main__":

    # run_tetris("APEX", rng=random.Random(42),auto_gen_path=True)
    # run_tetris("gpt-4o", rng=random.Random(42),auto_gen_path=True)
    # run_tetris("VLM_APEX", rng=random.Random(42),auto_gen_path=True)
    # run_tetris("CLIP", rng=random.Random(42),auto_gen_path=True)
    # run_tetris("VLM", rng=random.Random(42),auto_gen_path=True)
    # run_tetris("gpt-4o-mini", rng=random.Random(42),auto_gen_path=True)
    # run_tetris("o3-mini", rng=random.Random(42),auto_gen_path=True)
    for model in results.keys():
        for i in range(5):  # 5 trails
            save_path = f"experiments/tetris_expt/results/tetris_results/{model.replace('/','')}_{i}_tetris_game_history.json"
            res = run_tetris(model, save_path, rng=random.Random(42 + i))
    
            results[model]["final_score"].append(res["final_score"])
            results[model]["max_stack_height"].append(res["max_stack_height"])
            results[model]["holes"].append(res["holes"])
            results[model]["bumps"].append(res["bumps"])
            results[model]["height_delta_per_move"].append(res["height_delta_per_move"])
            results[model]["resp_time"].append(res["avg_resp_time"])
            with open("experiments/tetris_expt/results/tetris_results/tetris_eval_results.json", "w") as f:
                json.dump(results, f, indent=2)
