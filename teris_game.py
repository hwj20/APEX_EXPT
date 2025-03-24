import random

import pygame
from experiments.teris_game_agent import *
from experiments.utils.Tetris import *

pygame.init()
results = {
    "PGD": {"max_stack_height": [], "rounds_survived": [], "lines_cleared": []},
    "VLM+PGD": {"max_stack_height": [], "rounds_survived": [], "lines_cleared": []},
    "VLM": {"max_stack_height": [], "rounds_survived": [], "lines_cleared": []},
    "gpt-4o": {"max_stack_height": [], "rounds_survived": [], "lines_cleared": []},
    "gpt-4o-mini": {"max_stack_height": [], "rounds_survived": [], "lines_cleared": []},
}


def run_tetris(method, save_path="tetris_game_history_30_pgd.json"):
    # 运行游戏
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    tetris = Tetris(rng=random.Random(42))
    ag = LLM_Agent()

    game_history = []  # 存储游戏历史
    cnt = 0
    MAX_EPOCH = 30

    while tetris.running:
        if cnt < MAX_EPOCH:
            cnt += 1
        else:
            break
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                tetris.running = False

        # 获取当前游戏状态
        state = tetris.get_state()

        if state != tetris.previous_state and tetris.piece_y > 1:  # 只有状态变化时才执行
            tetris.render(screen)

            # 选择不同 AI 方案
            action_json = ''
            if method == 'PGD':
                pgd_results = tetris.apex_evaluate()
                action_json = ag.decide_move_pgd(state, pgd_results)
            elif method == "gpt-4o" or method == "gpt-4o-mini":
                action_json = ag.decide_move(state, method)
            elif method == 'VLM':
                image_path = tetris.capture_game_screen(screen)
                action_json = ag.vlm_decide_move(image_path)
            elif method == 'VLM_PGD':
                pgd_results = tetris.apex_evaluate()
                image_path = tetris.capture_game_screen(screen)
                action_json = ag.vlm_decide_move_pgd(image_path, pgd_results)

            # 确保 AI 不会返回 None
            if action_json is None:
                action_json = '{"move":"down","times":1}'

            # 记录当前状态
            game_history.append({
                "board": state["board"],
                "piece": state["piece"],
                "piece_x": state["piece_x"],
                "piece_y": state["piece_y"],
                "action": action_json
            })

            tetris.previous_state = state  # 更新之前的状态
            print(state)
            print(action_json)

            # 执行 AI 选择的操作
            tetris.step(action_json)

        # 处理重力
        tetris.gravity()
        tetris.render(screen)
        clock.tick(1)  # 控制游戏速度

    pygame.quit()

    # 游戏结束，保存所有历史状态
    with open(save_path, "w") as f:
        json.dump(game_history, f, indent=4)
    print(f"游戏历史已保存到 {save_path}")

    return

if __name__ == "__main__":
    run_tetris("PGD")


# for model in results.keys():
#     for i in range(5):  # 每个模型跑 5 盘
#         save_path = model + "_tetris_game_history.json"
#         run_tetris(model, save_path)
# max_height = run_tetris(model)["max_stack_height"]
# rounds = run_tetris(model)["rounds_survived"]
# lines = run_tetris(model)["lines_cleared"]
#
# results[model]["max_stack_height"].append(max_height)
# results[model]["rounds_survived"].append(rounds)
# results[model]["lines_cleared"].append(lines)


def if_you_want_to_play_game():
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
                    tetris.step("left")
                elif event.key == pygame.K_RIGHT:
                    tetris.step("right")
                elif event.key == pygame.K_UP:
                    tetris.step("rotate")
                elif event.key == pygame.K_DOWN:
                    tetris.step("down")

        tetris.gravity()
        tetris.render(screen)
        clock.tick(1)  # 控制游戏速度

    pygame.quit()
