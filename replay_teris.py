import json
import pygame
from experiments.utils.Tetris import *
def generate_tetris_action_diff_grid(method, history_file="tetris_game_history_30_pgd.json", output_path="tetris_action_diff_grid.png"):
    import json
    from PIL import Image
    import pygame

    output_path = method + "_" + output_path

    with open(history_file, "r") as f:
        game_history = json.load(f)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # tetris = Tetris()

    image_list = []

    for step in game_history[:15]:  # 每步生成2张图 → 最多拼30张
        # Step Before
        screen.fill(BLACK)
        tetris = Tetris(generate_new_piece=False)
        tetris.board = [row[:] for row in step["board"]]
        # v_map = {1:2,2:1,0:0}
        # for i in range(len(tetris.board)):
        #     for j in range(len(tetris.board[0])):
        #         tetris.board[i][j] = v_map[tetris.board[i][j]]
        tetris.current_piece = [row[:] for row in step["piece"]]
        tetris.piece_x = step["piece_x"]
        tetris.piece_y = step["piece_y"]
        tetris.score = step["score"]
        tetris.render(screen)

        pre_img = pygame.surfarray.array3d(pygame.display.get_surface())
        pre_img = pre_img.transpose([1, 0, 2])
        image_list.append(Image.fromarray(pre_img))

        tetris.gravity()
        # Step After (调用 step() 执行动作)
        tetris.step(step["action"])
        print(step['action'])
        # screen.fill(BLACK)
        tetris.render(screen)

        post_img = pygame.surfarray.array3d(pygame.display.get_surface())
        post_img = post_img.transpose([1, 0, 2])
        image_list.append(Image.fromarray(post_img))

    pygame.quit()

    # 拼图：5列 × 6行（前后图像交错排）
    cols = 6
    rows = 5  # 15 steps × 2 = 30 图像
    thumb_width = WIDTH
    thumb_height = HEIGHT
    grid_image = Image.new("RGB", (cols * thumb_width, rows * thumb_height))

    for idx, im in enumerate(image_list):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * thumb_height
        grid_image.paste(im, (x, y))

    grid_image.save(output_path)
    print(f"✅ Tetris action diff grid saved to {output_path}")

def generate_tetris_summary_grid(method,history_file="tetris_game_history_30_pgd.json", output_path="tetris_summary_grid.png"):
    output_path = method+"_"+output_path
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
    method = "APEX"
    save_path = "tetris_game_history_"
    save_type = ".json"
    save_path = save_path+method+save_type
    generate_tetris_action_diff_grid(method,save_path)
    # generate_tetris_summary_grid(method,save_path)
    # replay_tetris()
