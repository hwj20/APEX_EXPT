import numpy as np
import pygame
import random

# 游戏参数
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
COLS = SCREEN_WIDTH // BLOCK_SIZE
ROWS = SCREEN_HEIGHT // BLOCK_SIZE

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0), (128, 0, 128), (0, 255, 255)]

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
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_piece = self.new_piece()
        self.piece_x = COLS // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        self.running = True

    def new_piece(self):
        return random.choice(SHAPES), random.choice(COLORS)

    def valid_move(self, shape, x, y):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    new_x, new_y = x + j, y + i
                    if new_x < 0 or new_x >= COLS or new_y >= ROWS or (new_y >= 0 and self.board[new_y, new_x]):
                        return False
        return True

    def place_piece(self):
        shape, color = self.current_piece
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    self.board[self.piece_y + i, self.piece_x + j] = COLORS.index(color) + 1
        self.clear_rows()
        self.current_piece = self.new_piece()
        self.piece_x = COLS // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0
        if not self.valid_move(*self.current_piece, self.piece_x, self.piece_y):
            self.running = False

    def clear_rows(self):
        new_board = [row for row in self.board if not all(row)]
        while len(new_board) < ROWS:
            new_board.insert(0, np.zeros(COLS, dtype=int))
        self.board = np.array(new_board)

    def move_piece(self, dx, dy):
        if self.valid_move(self.current_piece[0], self.piece_x + dx, self.piece_y + dy):
            self.piece_x += dx
            self.piece_y += dy
        elif dy > 0:
            self.place_piece()

    def rotate_piece(self):
        rotated = list(zip(*self.current_piece[0][::-1]))
        if self.valid_move(rotated, self.piece_x, self.piece_y):
            self.current_piece = (rotated, self.current_piece[1])

    def draw_board(self):
        self.screen.fill(BLACK)
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, COLORS[cell - 1],
                                     (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        shape, color = self.current_piece
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, color,
                                     ((self.piece_x + j) * BLOCK_SIZE, (self.piece_y + i) * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_piece(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.move_piece(1, 0)
                    elif event.key == pygame.K_DOWN:
                        self.move_piece(0, 1)
                    elif event.key == pygame.K_UP:
                        self.rotate_piece()
            self.move_piece(0, 1)
            self.draw_board()
        pygame.quit()

if __name__ == "__main__":
    game = Tetris()
    game.run()