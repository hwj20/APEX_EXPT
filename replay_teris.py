import json
import pygame
from experiments.utils.Tetris import *


def replay_tetris(history_file="tetris_game_history.json"):
    with open(history_file, "r") as f:
        game_history = json.load(f)

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    tetris = Tetris()

    for step in game_history:
        screen.fill(BLACK)
        tetris.board = step["board"]
        tetris.current_piece = step["piece"]
        tetris.piece_x = step["piece_x"]
        tetris.piece_y = step["piece_y"]
        tetris.render(screen)
        pygame.display.flip()
        clock.tick(1)  # 控制回放速度

    pygame.quit()
    print("游戏回放完成！")


if __name__ == "__main__":
    replay_tetris()
