import pygame
import random
from experiments.teris_game_agent import *
import openai  # 假设我们用 OpenAI API 作为 LLM

# 初始化 Pygame
pygame.init()

# 游戏窗口大小
WIDTH, HEIGHT = 300, 600
BLOCK_SIZE = 30  # 每个方块的大小
COLUMNS = WIDTH // BLOCK_SIZE
ROWS = HEIGHT // BLOCK_SIZE
GRAVITY_SPEED = 1
# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# 俄罗斯方块形状
SHAPES = [
    [[1, 1, 1, 1]],  # I 形
    [[1, 1], [1, 1]],  # O 形
    [[1, 1, 1], [0, 1, 0]],  # T 形
    [[1, 1, 0], [0, 1, 1]],  # Z 形
    [[0, 1, 1], [1, 1, 0]],  # S 形
    [[1, 1, 1], [1, 0, 0]],  # L 形
    [[1, 1, 1], [0, 0, 1]]  # J 形
]


class Tetris:
    def __init__(self):
        self.board = [[0] * COLUMNS for _ in range(ROWS)]  # 10x20 的棋盘
        self.current_piece = self.new_piece()  # 当前活动方块
        self.piece_x = COLUMNS // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        self.running = True
        self.score = 0  # 计分系统！
        self.gravity_counter = 0  # 控制重力
        self.previous_state = None

    def new_piece(self):
        """生成一个新方块"""
        return random.choice(SHAPES)

    def move(self, dx, dy):
        """尝试移动方块，如果碰撞则不移动"""
        if not self.check_collision(dx, dy):
            self.piece_x += dx
            self.piece_y += dy

    def rotate(self):
        """旋转方块"""
        rotated = [list(row) for row in zip(*self.current_piece[::-1])]
        if not self.check_collision(0, 0, rotated):  # 旋转后不碰撞才允许
            self.current_piece = rotated

    def check_collision(self, dx=0, dy=0, piece=None):
        """检测是否会碰撞"""
        piece = piece or self.current_piece
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.piece_x + x + dx
                    new_y = self.piece_y + y + dy
                    if new_x < 0 or new_x >= COLUMNS or new_y >= ROWS or (new_y >= 0 and self.board[new_y][new_x]):
                        return True  # 碰撞了
        return False

    def has_landed(self):
        """检测当前方块是否落地"""
        return self.check_collision(dy=1)  # 看看下一行有没有空间

    def place_piece(self):
        """将方块固定到棋盘"""
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.piece_y + y][self.piece_x + x] = 1  # 记录到 board
        self.clear_lines()  # 试图消行
        self.current_piece = self.new_piece()  # 生成新方块
        self.piece_x = COLUMNS // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        if self.check_collision():  # 如果新方块一生成就碰撞，游戏结束
            self.running = False

    def clear_lines(self):
        """检查棋盘，清除满行，并计分"""
        new_board = [row for row in self.board if not all(row)]  # 只保留未满行的行
        lines_cleared = ROWS - len(new_board)
        self.score += lines_cleared * 100  # 每行 +100 分
        self.board = [[0] * COLUMNS for _ in range(lines_cleared)] + new_board  # 重新填充棋盘

    def step(self, action):
        """执行玩家或 LLM 的操作"""
        if action == "left":
            self.move(-1, 0)
        elif action == "right":
            self.move(1, 0)
        elif action == "rotate":
            self.rotate()
        elif action == "down":
            while not self.has_landed():
                self.piece_y += 1  # 直接到底
            self.place_piece()  # 落地后固定

    def gravity(self):
        """控制方块自动下落"""
        self.gravity_counter += 1
        if self.gravity_counter >= GRAVITY_SPEED:
            if self.has_landed():
                self.place_piece()
            else:
                self.move(0, 1)
            self.gravity_counter = 0

    def pgd_evaluate(self):
        """测试四种操作后的棋盘和得分情况（不影响当前游戏状态）"""
        scores = {}
        for action in ["left", "right", "rotate", "down"]:
            temp_game = Tetris()
            temp_game.board = [row[:] for row in self.board]
            temp_game.current_piece = [row[:] for row in self.current_piece]
            temp_game.piece_x = self.piece_x
            temp_game.piece_y = self.piece_y
            temp_game.score = self.score
            temp_game.step(action)
            scores[action] = (temp_game.score, temp_game.board)  # 返回得分和棋盘状态
        return scores

    def get_state(self):
        """返回当前棋盘和方块位置"""
        return {
            "board": self.board,
            "piece": self.current_piece,
            "piece_x": self.piece_x,
            "piece_y": self.piece_y
        }

    def render(self, screen):
        """渲染游戏画面"""
        screen.fill(BLACK)

        # 绘制固定的棋盘方块
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, BLUE, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # 绘制当前下落的方块
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, WHITE, (
                    (self.piece_x + x) * BLOCK_SIZE, (self.piece_y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # 显示分数
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()


# 运行游戏
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# clock = pygame.time.Clock()
# tetris = Tetris()
#
# while tetris.running:
#     screen.fill(BLACK)
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             tetris.running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 tetris.step("left")
#             elif event.key == pygame.K_RIGHT:
#                 tetris.step("right")
#             elif event.key == pygame.K_UP:
#                 tetris.step("rotate")
#             elif event.key == pygame.K_DOWN:
#                 tetris.step("down")
#
#     tetris.gravity()
#     tetris.render(screen)
#     clock.tick(1)  # 控制游戏速度
#
# pygame.quit()

# 运行游戏
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
tetris = Tetris()
ag = LLM_Agent()

while tetris.running:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            tetris.running = False

    # 获取 LLM 的决策
    state = tetris.get_state()
    if state != tetris.previous_state:  # 只有当状态发生变化时才去咨询 LLM
        action = ag.decide_move(state)
        tetris.previous_state = state  # 更新之前的状态
        print(state)
        print(action)
        tetris.step(action)

    tetris.gravity()
    tetris.render(screen)
    clock.tick(1)  # 控制游戏速度

pygame.quit()
