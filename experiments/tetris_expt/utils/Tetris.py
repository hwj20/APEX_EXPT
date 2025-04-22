import json
import re
import pygame
import random

from experiments.tetris_expt.utils.tetris_game_agent import strip_markdown

# Game Window
WIDTH, HEIGHT = 300, 600
BLOCK_SIZE = 30  # 每个方块的大小
COLUMNS = WIDTH // BLOCK_SIZE
ROWS = HEIGHT // BLOCK_SIZE
GRAVITY_SPEED = 1
# Color
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I 形
    [[1, 1], [1, 1]],  # O 形
    [[1, 1, 1], [0, 1, 0]],  # T 形
    [[1, 1, 0], [0, 1, 1]],  # Z 形
    [[0, 1, 1], [1, 1, 0]],  # S 形
    [[1, 1, 1], [1, 0, 0]],  # L 形
    [[1, 1, 1], [0, 0, 1]]  # J 形
]





LANDED = 2
CURRENT_PIECE = 1

class Tetris:
    def __init__(self, rng=None, generate_new_piece=True):
        self.generate_new_piece = generate_new_piece
        self.rng = rng or random.Random()  # 不会影响外部 random
        self.board = [[0] * COLUMNS for _ in range(ROWS)]  # 10x20 的棋盘
        if self.generate_new_piece:
            self.current_piece = self.new_piece()  # 当前活动方块
            self.piece_x = COLUMNS // 2 - len(self.current_piece[0]) // 2
            self.piece_y = 0
        self.running = True
        self.score = 0  # 计分系统！
        self.gravity_counter = 0  # 控制重力
        self.previous_state = None
    def final_evaluation(self):
        """Evaluate final score, max height, holes, bumps, and height change per move"""
        def get_column_heights(board):
            heights = [0] * COLUMNS
            for x in range(COLUMNS):
                for y in range(ROWS):
                    if board[y][x]:
                        heights[x] = ROWS - y
                        break
            return heights

        def count_holes(board):
            holes = 0
            for col in zip(*board):
                found_block = False
                for cell in col:
                    if cell:
                        found_block = True
                    elif found_block:
                        holes += 1
            return holes

        def bumpiness(heights):
            return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

        heights = get_column_heights(self.board)
        max_stack_height = max(heights)
        hole_count = count_holes(self.board)
        bump = bumpiness(heights)
        height_delta = max_stack_height / max(1, self.gravity_counter)

        return {
            "final_score": self.score,
            "max_stack_height": max_stack_height,
            "holes": hole_count,
            "bumps": bump,
            "height_delta_per_move": round(height_delta, 2)
        }

    def new_piece(self):
        """生成一个新方块"""
        return self.rng.choice(SHAPES)

    def move(self, dx, dy):
        """尝试移动方块，如果碰撞则不移动"""
        if not self.check_collision(dx, dy):
            self.piece_x += dx
            self.piece_y += dy
            return True
        else:
            # print("not valid move")
            return False

    def rotate(self):
        """旋转方块"""
        rotated = [list(row) for row in zip(*self.current_piece[::-1])]
        if not self.check_collision(0, 0, rotated):  # 旋转后不碰撞才允许
            self.current_piece = rotated
            return True
        else:
            return False

    def check_collision(self, dx=0, dy=0, piece=None):
        """检测是否会碰撞"""
        piece = piece or self.current_piece
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.piece_x + x + dx
                    new_y = self.piece_y + y + dy
                    if new_x < 0 or new_x >= COLUMNS or new_y >= ROWS or (new_y >= 0 and self.board[new_y][new_x] == LANDED):
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
                    self.board[self.piece_y + y][self.piece_x + x] = LANDED  # 记录到 board
        self.clear_lines()  # 试图消行
        if self.generate_new_piece:
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
        return lines_cleared

    def step(self, action_json):
        """解析 LLM 代理的 JSON 并执行动作"""
        success = True
        action_json = strip_markdown(action_json)
        # print(action_json)
        action_data_all = json.loads(action_json)  # 解析 JSON 格式的指令
        if not isinstance(action_data_all, list):
            action_data_all = [action_data_all]
        for action_data in action_data_all:
            action = action_data.get("move", "down")
            times = action_data.get("times", 1)  # 默认执行 1 次

            for _ in range(times):
                if action == "left":
                    if not self.move(-1, 0):
                        success = False
                elif action == "right":
                    if not self.move(1, 0):
                        success = False
                elif action == "rotate":
                    if not self.rotate():
                        success = False
                elif action == "down":
                    while not self.has_landed():
                        self.piece_y += 1  # 直接到底
        return success

    def gravity(self):
        """控制方块自动下落"""
        self.gravity_counter += 1
        if self.gravity_counter >= GRAVITY_SPEED:
            if self.has_landed():
                self.place_piece()
            else:
                self.move(0, 1)
            self.gravity_counter = 0

    def extract_active_rows(self, board):
        """去除全零行，仅返回堆积部分"""
        return [row for row in board if any(row)]

    def apex_evaluate(self):
        """Evaluate multiple placements of current piece with physics-based metrics."""
        from copy import deepcopy

        def get_column_heights(board):
            heights = [0] * COLUMNS
            for x in range(COLUMNS):
                for y in range(ROWS):
                    if board[y][x]:
                        heights[x] = ROWS - y
                        break
            return heights

        def count_holes(board):
            holes = 0
            for col in zip(*board):
                found_block = False
                for cell in col:
                    if cell:
                        found_block = True
                    elif found_block:
                        holes += 1
            return holes

        def bumpiness(heights):
            return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

        results = {}

        # Simulation for different rotations
        for rotation_times in range(4):
            for dx in range(-COLUMNS, COLUMNS):
                # Deepcopy game state
                temp_game = Tetris(rng=random.Random(999), generate_new_piece=False)
                temp_game.board = deepcopy(self.board)
                temp_game.current_piece = deepcopy(self.current_piece)
                temp_game.piece_x = self.piece_x
                temp_game.piece_y = self.piece_y
                temp_game.score = self.score

                # Try rotation
                for _ in range(rotation_times):
                    temp_game.step(json.dumps({"move": "rotate", "times": 1}))

                # Try horizontal move
                action = "right" if dx > 0 else "left"
                not_valid = False
                for _ in range(abs(dx)):
                    if not temp_game.step(json.dumps({"move": action, "times": 1})):
                        not_valid = True
                        break

                # Check if placement is valid
                # if not temp_game.valid_position():
                #     continue
                if not_valid:
                    continue

                # Drop piece
                while not temp_game.has_landed():
                    temp_game.piece_y += 1
                temp_game.place_piece()

                # Clear lines and compute post-state
                cleared_lines = temp_game.clear_lines()
                final_board = temp_game.board
                heights = get_column_heights(final_board)
                stack_height = max(heights)
                hole_count = count_holes(final_board)
                surface_bump = bumpiness(heights)

                results[str([("rotate", rotation_times), (action, abs(dx))])] = {
                    "cleared_lines": cleared_lines,
                    "highest_stack": stack_height,
                    "holes": hole_count,
                    "bumpiness": surface_bump,
                    "column_heights": heights,
                }

        return results

    def valid_position(self, piece=None, offset_x=None, offset_y=None):
        """Check whether the piece is in a valid position on the board."""
        piece = piece if piece is not None else self.current_piece
        x_offset = offset_x if offset_x is not None else self.piece_x
        y_offset = offset_y if offset_y is not None else self.piece_y

        for row_idx, row in enumerate(piece):
            for col_idx, cell in enumerate(row):
                if cell:
                    board_x = x_offset + col_idx
                    board_y = y_offset + row_idx
                    # 越界检测
                    if board_x < 0 or board_x >= COLUMNS or board_y < 0 or board_y >= ROWS:
                        return False
                    # 碰撞检测
                    if self.board[board_y][board_x]:
                        return False
        return True

    def get_state(self):
        """返回当前棋盘和方块位置，方块标记为 1"""
        board_with_piece = [row[:] for row in self.board]  # 复制棋盘
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    board_y = self.piece_y + y
                    board_x = self.piece_x + x
                    if 0 <= board_y < ROWS and 0 <= board_x < COLUMNS:
                        board_with_piece[board_y][board_x] = CURRENT_PIECE  # 标记当前活动方块

        return {
            "board": board_with_piece,
            "piece": self.current_piece,
            "piece_x": self.piece_x,
            "piece_y": self.piece_y,
            "score": self.score
        }

    def render(self, screen):
        """渲染游戏画面"""
        screen.fill(BLACK)

        # 绘制固定的棋盘方块
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell == LANDED:
                    pygame.draw.rect(screen, BLUE, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # 绘制当前下落的方块
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell == CURRENT_PIECE:
                    pygame.draw.rect(screen, WHITE, (
                        (self.piece_x + x) * BLOCK_SIZE, (self.piece_y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # 显示分数
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def capture_game_screen(self, screen):
        """截取当前游戏画面，转换成 VLM 可读格式"""
        pygame.image.save(screen, "screenshot.png")  # 存为 PNG
        return "screenshot.png"  # 返回文件路径
