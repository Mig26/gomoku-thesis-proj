import operator
import time
import pygame
import testai
import ai
import random
import stats


class GomokuGame:
    def __init__(self, values):
        self.GRID_SIZE = values[1]
        self.WIDTH = self.HEIGHT = self.GRID_SIZE * values[0]
        self.CELL_SIZE = self.WIDTH // self.GRID_SIZE
        self.P1COL = values[2]
        self.P2COL = values[3]
        self.BOARD_COL = values[4]
        self.LINE_COL = values[5]
        self.SLEEP_BEFORE_END = values[6]
        self.board = [[0] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.winning_cells = []
        self.current_game = 0
        self.last_round = False
        self.ai_delay = False


class Player:
    def __init__(self, player_type, player_id):
        self.TYPE = str(player_type)
        self.ID = int(player_id)
        self.moves = 0
        self.wins = 0
        self.losses = 0
        self.score = 0
        self.sum_score = 0
        self.avg_score = 0
        self.all_moves = []
        self.avg_moves = 0
        self.weighed_scores = []
        self.win_rate = 0
        if self.ID == 2:
            self.ai = ai.GomokuAI()

    def set_player(self, player_type, player_id):
        self.TYPE = str(player_type)
        self.ID = int(player_id)
        if self.ID == 2:
            self.ai = ai.GomokuAI()
        print("Set player", self.ID, "to", self.TYPE)

    def get_player(self):
        return self

    def calculate_score(self, max_score, is_winner, game_number):
        if max_score > 0:
            if is_winner:
                self.score = max_score - self.moves
            else:
                self.score = -max_score + self.moves
            weighed_score = self.score / max_score
            self.weighed_scores.append(weighed_score)
        else:
            self.score = 0
            self.weighed_scores.append(0)
        self.sum_score += self.score
        self.avg_score = self.sum_score / game_number
        self.all_moves.append(self.moves)
        self.avg_moves = sum(self.all_moves) / len(self.all_moves)

    def calculate_win_rate(self, rounds):
        self.win_rate = self.wins / rounds

    def reset_score(self):
        self.score = 0
        self.moves = 0

    def reset_all_stats(self):
        self.moves = 0
        self.wins = 0
        self.losses = 0
        self.score = 0
        self.sum_score = 0
        self.avg_score = 0
        self.all_moves = []
        self.avg_moves = 0


# Set default player types. Can be changed on runtime
player1 = Player("Human", 0)
player2 = Player("AI", 1)
players = [player1, player2]


def reset_player_stats():
    for i in range(len(players)):
        players[i].reset_score()


def update_player_stats(instance, winning_player):
    global players
    if winning_player > -1: # run if game was not a tie
        for i in range(len(players)):
            if i == winning_player:
                players[i].wins += 1
                is_winner = True
                # players[i].calculate_score(instance.GRID_SIZE**2, True, instance.current_game)
            else:
                players[i].losses += 1
                is_winner = False
                # players[i].calculate_score(instance.GRID_SIZE**2, False, instance.current_game)
            players[i].calculate_score(instance.GRID_SIZE ** 2, is_winner, instance.current_game)
            if instance.last_round:
                players[i].calculate_win_rate(instance.current_game)
    else:
        for i in range(len(players)):
            players[i].calculate_score(0, False, instance.current_game)
    stats.log_win(players)
    reset_player_stats()
    if instance.last_round:
        stats.log_message(f"\nStatistics:\n{players[0].TYPE} {players[0].ID}:\nwins: {players[0].wins} - win rate: {players[0].win_rate} - average score: {players[0].avg_score} - weighed score: {sum(players[0].weighed_scores)/len(players[0].weighed_scores)} - average moves: {players[0].avg_moves}.\n"
                          f"{players[1].TYPE} {players[1].ID}:\nwins: {players[1].wins} - win rate: {players[1].win_rate} - average score: {players[1].avg_score} - weighed score: {sum(players[1].weighed_scores)/len(players[1].weighed_scores)} - average moves: {players[1].avg_moves}.")
        players[0].reset_all_stats()
        players[1].reset_all_stats()


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
    for drow, dcol in directions:
        winning_cells = [(row, col)]
        winning_direction = ()
        count = 1
        # positive direction
        for i in range(1, 5):
            row_, col_ = row + i * drow, col + i * dcol
            if 0 <= row_ < instance.GRID_SIZE and 0 <= col_ < instance.GRID_SIZE and instance.board[row_][col_] == player:
                count += 1
                winning_cells.append((row_, col_))
                winning_direction = [(drow, dcol)]
            else:
                break
        # negative direction
        for i in range(1, 5):
            row_, col_ = row - i * drow, col - i * dcol
            if 0 <= row_ < instance.GRID_SIZE and 0 <= col_ < instance.GRID_SIZE and instance.board[row_][col_] == player:
                count += 1
                winning_cells.append((row_, col_))
                winning_direction = (drow, dcol)
            else:
                break
        if count >= 5:  # Victory condition
            match winning_direction:    # sort the array so that a strike can be drawn correctly
                case (1, 0):
                    winning_cells.sort()
                case(0, 1):
                    winning_cells.sort(key=lambda i: i[1])
                case(1, 1):
                    winning_cells.sort(key=operator.itemgetter(0, 1))
                case(1, -1):
                    winning_cells.sort(key=operator.itemgetter(0, 1), reverse=True)
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
    while running:
        if not check_board_full(instance):
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
                            players[current_player - 1].moves += 1
                            if check_win(row, col, current_player, instance):
                                victory_text = f"Player {current_player} wins!"
                                running = False
                            else:
                                # Switch player if neither player have won
                                current_player = 3 - current_player
            # AI move
            elif players[current_player-1].TYPE == "AI" and not testai.check_game_over(instance):
                if instance.ai_delay:
                    time.sleep(random.uniform(0.25, 1.0))   # randomize ai "thinking" time
                ai_row, ai_col = testai.ai_move(instance, players[current_player-1].ID)
                testai.make_move((ai_row, ai_col), current_player, instance)
                players[current_player-1].moves += 1
                if check_win(ai_row, ai_col, current_player, instance):
                    victory_text = f"AI {players[current_player-1].ID} wins!"
                    running = False
                else:
                    current_player = 3 - current_player
            draw_board(instance)
            pygame.display.flip()
            window_name = "Gomoku -- Player " + str(current_player)
            pygame.display.set_caption(window_name)
        else:
            victory_text = "TIE"
            current_player = -1
            running = False

    # End game
    stats.log_message(victory_text)
    pygame.display.set_caption("Gomoku -- " + victory_text)
    update_player_stats(instance, current_player-1)
    time.sleep(instance.SLEEP_BEFORE_END)
    reset_game(instance)


pygame.quit()
