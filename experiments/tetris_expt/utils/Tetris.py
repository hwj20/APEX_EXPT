import json
import re
import pygame
import random

from experiments.tetris_expt.utils.tetris_game_agent import strip_markdown

# Game Window
WIDTH, HEIGHT = 300, 600
BLOCK_SIZE = 30
COLUMNS = WIDTH // BLOCK_SIZE
ROWS = HEIGHT // BLOCK_SIZE
GRAVITY_SPEED = 1
# Color
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]]  # J
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
        """generate a new piece"""
        return self.rng.choice(SHAPES)

    def move(self, dx, dy):
        """move a piece"""
        if not self.check_collision(dx, dy):
            self.piece_x += dx
            self.piece_y += dy
            return True
        else:
            # print("not valid move")
            return False

    def rotate(self):
        """rotate a piece"""
        rotated = [list(row) for row in zip(*self.current_piece[::-1])]
        if not self.check_collision(0, 0, rotated):  # no collision after rotation
            self.current_piece = rotated
            return True
        else:
            return False

    def check_collision(self, dx=0, dy=0, piece=None):
        """check collision"""
        piece = piece or self.current_piece
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.piece_x + x + dx
                    new_y = self.piece_y + y + dy
                    if new_x < 0 or new_x >= COLUMNS or new_y >= ROWS or (new_y >= 0 and self.board[new_y][new_x] == LANDED):
                        return True  # collision
        return False

    def has_landed(self):
        """check if the piece landed"""
        return self.check_collision(dy=1)  # to see if it can drop

    def place_piece(self):
        """fixed a piece in board when it landed"""
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    self.board[self.piece_y + y][self.piece_x + x] = LANDED  # record to board
        self.clear_lines()  # try to clear lines
        if self.generate_new_piece:
            self.current_piece = self.new_piece()  # generate new piece
        self.piece_x = COLUMNS // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        if self.check_collision():  # game ends when collision found in generating new piece
            self.running = False

    def clear_lines(self):
        """clear lines, record scores"""
        new_board = [row for row in self.board if not all(row)]
        lines_cleared = ROWS - len(new_board)
        self.score += lines_cleared * 100  # + 100 pt/line
        self.board = [[0] * COLUMNS for _ in range(lines_cleared)] + new_board
        return lines_cleared

    # control policy
    def step(self, action_json):
        success = True
        action_json = strip_markdown(action_json)
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
        """move down each time when gravity_counter=GRAVITY_SPEED"""
        self.gravity_counter += 1
        if self.gravity_counter >= GRAVITY_SPEED:
            if self.has_landed():
                self.place_piece()
            else:
                self.move(0, 1)
            self.gravity_counter = 0

    def extract_active_rows(self, board):
        """return non-zero rows"""
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
                    # Boundary check
                    if board_x < 0 or board_x >= COLUMNS or board_y < 0 or board_y >= ROWS:
                        return False
                    # Collision check
                    if self.board[board_y][board_x]:
                        return False
        return True

    def get_state(self):
        """return current states"""
        board_with_piece = [row[:] for row in self.board]  # copy board
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    board_y = self.piece_y + y
                    board_x = self.piece_x + x
                    if 0 <= board_y < ROWS and 0 <= board_x < COLUMNS:
                        board_with_piece[board_y][board_x] = CURRENT_PIECE  # mark current piece

        return {
            "board": board_with_piece,
            "piece": self.current_piece,
            "piece_x": self.piece_x,
            "piece_y": self.piece_y,
            "score": self.score
        }

    def render(self, screen):
        """render game screen"""
        screen.fill(BLACK)

        # draw fixed pieces
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell == LANDED:
                    pygame.draw.rect(screen, BLUE, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # draw dropping pieces
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell == CURRENT_PIECE:
                    pygame.draw.rect(screen, WHITE, (
                        (self.piece_x + x) * BLOCK_SIZE, (self.piece_y + y) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # show scores
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def capture_game_screen(self, screen):
        """screenshot for VLM"""
        pygame.image.save(screen, "screenshot.png")
        return "screenshot.png"
