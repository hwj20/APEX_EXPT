import json
import pygame
import random

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
        random.seed(42)
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

    def step(self, action_json):
        """解析 LLM 代理的 JSON 并执行动作"""
        action_data = json.loads(action_json)  # 解析 JSON 格式的指令
        action = action_data.get("move", "down")
        times = action_data.get("times", 1)  # 默认执行 1 次

        for _ in range(times):
            if action == "left":
                self.move(-1, 0)
            elif action == "right":
                self.move(1, 0)
            elif action == "rotate":
                self.rotate()
            elif action == "down":
                while not self.has_landed():
                    self.piece_y += 1  # 直接到底

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

    def pgd_evaluate(self):
        """模拟多种操作的最终状态，包括落地位置、消行信息、最高堆积高度，仅返回堆积区域"""
        scores = {}
        actions = []
        for k in range(1, COLUMNS):  # 从左移1格到最大，再从右移1格到最大
            if self.piece_x - k >= 0:
                actions.append(("left", k))
            if self.piece_x + k < COLUMNS:
                actions.append(("right", k))
        for r in range(3):  # 最多旋转3次
            actions.append(("rotate", r + 1))
        actions.append(("down", 1))  # 直接下落

        for action, times in actions:
            temp_game = Tetris()
            temp_game.board = [row[:] for row in self.board]  # 复制棋盘
            temp_game.current_piece = [row[:] for row in self.current_piece]  # 复制方块形状
            temp_game.piece_x = self.piece_x
            temp_game.piece_y = self.piece_y
            temp_game.score = self.score

            # 执行移动/旋转操作
            for _ in range(times):
                temp_game.step(json.dumps({"move": action, "times": 1}))

            # 让方块掉落到底
            while not temp_game.has_landed():
                temp_game.piece_y += 1
            temp_game.place_piece()
            cleared_lines = temp_game.clear_lines()
            final_board = temp_game.board

            # 计算最高堆积点
            # highest_stack = ROWS - max(y for y, row in enumerate(final_board) if any(row)) if any(final_board) else 0

            # 仅返回堆积部分
            active_rows = self.extract_active_rows(final_board)
            highest_stack = len(active_rows)

            scores[(action, times)] = {
                "score": temp_game.score,
                # "cleared_lines": cleared_lines,
                # "final_board": final_board,
                "highest_stack": highest_stack,
                "active_rows": active_rows
            }
        return scores

    def get_state(self):
        """返回当前棋盘和方块位置，方块标记为 2"""
        board_with_piece = [row[:] for row in self.board]  # 复制棋盘
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    board_y = self.piece_y + y
                    board_x = self.piece_x + x
                    if 0 <= board_y < ROWS and 0 <= board_x < COLUMNS:
                        board_with_piece[board_y][board_x] = 2  # 标记当前活动方块

        return {
            "board": board_with_piece,
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
    def capture_game_screen(self,screen):
        """截取当前游戏画面，转换成 VLM 可读格式"""
        pygame.image.save(screen, "screenshot.png")  # 存为 PNG
        return "screenshot.png"  # 返回文件路径

