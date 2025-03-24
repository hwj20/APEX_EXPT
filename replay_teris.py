import json
import pygame
from experiments.utils.Tetris import *

def generate_tetris_summary_grid(history_file="tetris_game_history_30_pgd.json", output_path="tetris_summary_grid.png"):
    import os
    from PIL import Image

    with open(history_file, "r") as f:
        game_history = json.load(f)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    tetris = Tetris()

    image_list = []
    for step in game_history[:20]:
        screen.fill(BLACK)
        tetris.board = step["board"]
        tetris.current_piece = step["piece"]
        tetris.piece_x = step["piece_x"]
        tetris.piece_y = step["piece_y"]
        tetris.render(screen)

        # 截图存到内存
        screenshot = pygame.surfarray.array3d(pygame.display.get_surface())
        screenshot = screenshot.transpose([1, 0, 2])  # 变成 HWC 格式
        image = Image.fromarray(screenshot)
        image_list.append(image)

    pygame.quit()

    # 拼图：5列4行
    cols = 5
    rows = 3
    thumb_width = WIDTH
    thumb_height = HEIGHT
    grid_image = Image.new("RGB", (cols * thumb_width, rows * thumb_height))

    for idx, im in enumerate(image_list):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * thumb_height
        grid_image.paste(im, (x, y))

    grid_image.save(output_path)
    print(f"✅ Tetris summary saved to {output_path}")
def replay_tetris(history_file="tetris_game_history_30_mini.json"):
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
    generate_tetris_summary_grid()
    # replay_tetris()
