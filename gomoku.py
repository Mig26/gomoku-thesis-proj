import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 15
CELL_SIZE = WIDTH // GRID_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BEIGE = (232, 220, 202)
LINE_COLOR = (169, 169, 169)

# Initialize the game board
board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

# Initialize Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Gomoku')

# Function to draw the game board
def draw_board():
    screen.fill(BEIGE)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
            if board[row][col] == 1:
                pygame.draw.circle(screen, BLACK, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)
            elif board[row][col] == 2:
                pygame.draw.circle(screen, WHITE, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 5)

# Function to check for a win
def check_win(row, col, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for i in range(1, 5):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and board[r][c] == player:
                count += 1
            else:
                break
        for i in range(1, 5):
            r, c = row - i * dr, col - i * dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and board[r][c] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
    return False

# Main game loop
current_player = 1  # Player 1 starts
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            col = x // CELL_SIZE
            row = y // CELL_SIZE
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and board[row][col] == 0:
                board[row][col] = current_player
                if check_win(row, col, current_player):
                    print(f"Player {current_player} wins!")
                    running = False
                else:
                    current_player = 3 - current_player  # Switch player

    draw_board()
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()