import time
import pygame
import testai
import random
import stats


class GomokuGame:
    def __init__(self, values):
        self.WIDTH = values[0]
        self.HEIGHT = values[1]
        self.GRID_SIZE = values[2]
        self.CELL_SIZE = self.WIDTH // self.GRID_SIZE
        self.P1COL = values[3]
        self.P2COL = values[4]
        self.BOARD_COL = values[5]
        self.LINE_COL = values[6]
        self.SLEEP_BEFORE_END = values[7]
        self.board = [[0] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.winning_cells = []


class Player:
    def __init__(self, player_type, player_id):
        self.TYPE = str(player_type)
        self.ID = int(player_id)
        self.moves = 0
        self.wins = 0
        self.losses = 0
        self.score = 0

    def set_player(self, player_type, player_id):
        self.TYPE = str(player_type)
        self.ID = int(player_id)
        print("Set player", self.ID, "to", self.TYPE)

    def get_player(self):
        return self


player1 = Player("Human", 0)
player2 = Player("AI", 1)
players = [player1, player2]


def reset_player_stats():
    for i in range(len(players)):
        players[i].moves = 0
        players[i].score = 0


def update_player_stats(instance, winning_player):
    global players
    for i in range(len(players)):
        if i == winning_player:
            players[i].wins += 1
            players[i].score = instance.GRID_SIZE**2 - players[i].moves
        else:
            players[i].losses += 1
            players[i].score = -(instance.GRID_SIZE**2) + players[i].moves
    stats.log_win(players)
    reset_player_stats()


def set_players(_players):
    global players
    players = _players


window_name = "Gomoku"
victory_text = ""

current_player = 1


# Function to draw the game board
def draw_board(instance):
    instance.screen.fill(instance.BOARD_COL)
    cell_size = instance.CELL_SIZE
    for row in range(instance.GRID_SIZE):
        for col in range(instance.GRID_SIZE):
            pygame.draw.rect(instance.screen, instance.LINE_COL, (col * cell_size, row * cell_size, cell_size, cell_size), 1)
            if instance.board[row][col] == 1:
                pygame.draw.circle(instance.screen, instance.P1COL, (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2), cell_size // 2 - 5)
            elif instance.board[row][col] == 2:
                pygame.draw.circle(instance.screen, instance.P2COL, (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2), cell_size // 2 - 5)
    # Draw the winning line
    if instance.winning_cells:
        start_row, start_col = instance.winning_cells[0]
        end_row, end_col = instance.winning_cells[-1]
        pygame.draw.line(instance.screen, (0, 255, 0),
                         (start_col * cell_size + cell_size // 2, start_row * cell_size + cell_size // 2),
                         (end_col * cell_size + cell_size // 2, end_row * cell_size + cell_size // 2), 5)


def reset_game(instance):
    global current_player
    instance.board = [[0] * instance.GRID_SIZE for _ in range(instance.GRID_SIZE)]
    current_player = 1


def check_win(row, col, player, instance):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    winning_cells = [(row, col)]
    for dr, dc in directions:
        count = 1
        for i in range(1, 5):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < instance.GRID_SIZE and 0 <= c < instance.GRID_SIZE and instance.board[r][c] == player:
                count += 1
                winning_cells.append((r, c))
            else:
                break
        for i in range(1, 5):
            r, c = row - i * dr, col - i * dc
            if 0 <= r < instance.GRID_SIZE and 0 <= c < instance.GRID_SIZE and instance.board[r][c] == player:
                count += 1
                winning_cells.append((r, c))
            else:
                break
        if count >= 5:  # Victory condition
            instance.winning_cells = winning_cells
            return True
    return False


def check_board_full(instance):
    board = instance.board
    grid_size = instance.GRID_SIZE
    for row in range(grid_size):
        for col in range(grid_size):
            if board[row][col] == 0:
                return False
    return True


def run(instance):
    # Main game loop
    global window_name, victory_text, current_player
    # Initialize Pygame
    pygame.display.set_icon(pygame.image.load('res/ico.png'))
    pygame.init()
    pygame.display.set_caption(window_name)
    instance.winning_cells = []
    running = True
    while running and not check_board_full(instance):
        # Human move
        if players[current_player-1].TYPE == "Human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos
                    col = x // instance.CELL_SIZE
                    row = y // instance.CELL_SIZE
                    if instance.GRID_SIZE > row >= 0 == instance.board[row][col] and 0 <= col < instance.GRID_SIZE:
                        instance.board[row][col] = current_player
                        if check_win(row, col, current_player, instance):
                            victory_text = f"Player {current_player} wins!"
                            print(victory_text)
                            running = False
                        else:
                            # Switch player if neither player have won
                            current_player = 3 - current_player
        # AI move
        elif players[current_player-1].TYPE == "AI" and not testai.check_game_over(instance):
            time.sleep(random.uniform(0.25, 1.0))   # randomize ai "thinking" time
            ai_row, ai_col = testai.ai_move(instance, players[current_player-1].ID)
            testai.make_move((ai_row, ai_col), current_player, instance)
            players[current_player-1].moves += 1
            if check_win(ai_row, ai_col, current_player, instance):
                victory_text = f"AI {players[current_player-1].ID} wins!"
                print(victory_text)
                running = False
            else:
                current_player = 3 - current_player
        draw_board(instance)
        pygame.display.flip()
        window_name = "Gomoku -- Player " + str(current_player)
        pygame.display.set_caption(window_name)

    # End game
    pygame.display.set_caption("Gomoku -- " + victory_text)
    update_player_stats(instance, current_player-1)
    time.sleep(instance.SLEEP_BEFORE_END)
    reset_game(instance)


pygame.quit()
