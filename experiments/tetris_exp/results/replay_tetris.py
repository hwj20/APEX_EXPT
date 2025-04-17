from experiments.tetris_exp.utils.Tetris import *
import json
from PIL import Image
import pygame


def generate_tetris_action_diff_grid(method, history_file="tetris_game_history_30_pgd.json",
                                     output_path="tetris_action_diff_grid.png"):
    output_path = method + "_" + output_path

    with open(history_file, "r") as f:
        game_history = json.load(f)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    image_list = []

    for step in game_history:
        # Step Before
        screen.fill(BLACK)
        tetris = Tetris(generate_new_piece=False)
        tetris.board = [row[:] for row in step["board"]]
        tetris.current_piece = [row[:] for row in step["piece"]]
        tetris.piece_x = step["piece_x"]
        tetris.piece_y = step["piece_y"]
        tetris.score = step["score"]
        tetris.render(screen)

        pre_img = pygame.surfarray.array3d(pygame.display.get_surface()).transpose([1, 0, 2])
        image_list.append(Image.fromarray(pre_img))

        tetris.gravity()
        # Step After
        tetris.step(step["action"])
        tetris.render(screen)

        post_img = pygame.surfarray.array3d(pygame.display.get_surface()).transpose([1, 0, 2])
        image_list.append(Image.fromarray(post_img))

    pygame.quit()

    # 拼图：6列 × 3行
    cols = 6
    rows = 3
    thumb_width = WIDTH
    thumb_height = HEIGHT
    grid_image = Image.new("RGB", (cols * thumb_width, rows * thumb_height))

    for idx, im in enumerate(image_list):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * thumb_height
        grid_image.paste(im, (x, y))

    # 保存静态网格图
    grid_image.save(output_path)
    print(f"✅ Tetris action diff grid saved to {output_path}")

    # 生成 GIF 动图
    gif_path = method + "_action_diff.gif"
    image_list[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=image_list[1:],
        duration=500,  # 毫秒
        loop=0         # 无限循环
    )
    print(f"✅ Tetris action diff GIF saved to {gif_path}")


def generate_tetris_summary_grid(method, history_file="tetris_game_history_30_pgd.json",
                                 output_path="tetris_summary_grid.png"):
    output_path = method + "_" + output_path
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
        tetris.score = step['score']
        tetris.render(screen)

        screenshot = pygame.surfarray.array3d(pygame.display.get_surface()).transpose([1, 0, 2])
        image_list.append(Image.fromarray(screenshot))

    pygame.quit()

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
        clock.tick(1)

    pygame.quit()
    print("游戏回放完成！")


if __name__ == "__main__":
    method = "gpt-4o-mini"
    save_path = "../visual_demo/demo_"
    save_type = ".json"
    save_path = save_path + method + str(30) + save_type
    generate_tetris_action_diff_grid(method, save_path)
