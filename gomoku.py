import operator
import time
import pygame
import testai
import ai
import random
import stats
import numpy as np
import filereader


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
        self.score_loss = []
        self.weighed_moves = []
        self.move_loss = []
        self.final_move_scores = []
        self.final_move_loss = []
        self.win_rate = 0
        self.ai = ai.GomokuAI()
        self.final_action = None

    def set_player(self, player_type, player_id):
        self.TYPE = str(player_type)
        self.ID = int(player_id)
        #if self.ID == 2:
            #self.ai = ai.GomokuAI()
        print("Set player", self.ID, "to", self.TYPE)

    def get_player(self):
        return self

    def calculate_score(self, max_score, is_winner, game_number):
        if max_score > 0:
            if is_winner:
                self.score = (max_score - self.moves) / max_score
            else:
                self.score = -((max_score - self.moves) / max_score)
            # weighed_score = self.score / max_score
            self.weighed_scores.append(self.score)
        else:
            self.score = 0
            self.weighed_scores.append(0)
        print(f"score: {self.score}")
        self.sum_score += self.score
        self.avg_score = self.sum_score / game_number
        self.all_moves.append(self.moves)
        self.avg_moves = sum(self.all_moves) / len(self.all_moves)

    def calculate_win_rate(self, rounds):
        self.win_rate = self.wins / rounds

    def reset_score(self):
        self.score = 0
        self.moves = 0
        self.weighed_moves = []
        self.move_loss = []

    def reset_all_stats(self):
        self.moves = 0
        self.wins = 0
        self.losses = 0
        self.score = 0
        self.sum_score = 0
        self.avg_score = 0
        self.weighed_scores = []
        self.score_loss = []
        self.all_moves = []
        self.weighed_moves = []
        self.move_loss = []
        self.final_move_scores = []
        self.final_move_loss = []
        self.avg_moves = 0


# Set default player types. Can be changed on runtime
player1 = Player("Human", 0)
player2 = Player("MM-AI", 1)
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
            else:
                players[i].losses += 1
                is_winner = False
            players[i].calculate_score(instance.GRID_SIZE ** 2, is_winner, instance.current_game)
            if instance.last_round:
                players[i].calculate_win_rate(instance.current_game)
    else:
        for i in range(len(players)):
            players[i].calculate_score(0, False, instance.current_game)
    stats.log_win(players)
    if instance.last_round:
        stats.log_message(f"\nStatistics:\n{players[0].TYPE} {players[0].ID}:\nwins: {players[0].wins} - win rate: {players[0].win_rate} - average score: {players[0].avg_score} - weighed score: {sum(players[0].weighed_scores)/len(players[0].weighed_scores)} - average moves: {players[0].avg_moves}.\n"
                          f"{players[1].TYPE} {players[1].ID}:\nwins: {players[1].wins} - win rate: {players[1].win_rate} - average score: {players[1].avg_score} - weighed score: {sum(players[1].weighed_scores)/len(players[1].weighed_scores)} - average moves: {players[1].avg_moves}.")


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


def calculate_score(board: tuple, board_size=15):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    score_board = filereader.load_scores("consts.json")
    scored_board = np.zeros((board_size, board_size))
    for row in range(len(board[0])):
        for col in range(len(board[1])):
            adjacent_tiles = {}
            tiles = {}
            if board[row][col] == 0:
                for i in range(len(directions)):
                    forward = []
                    for j in range(5):
                        try:
                            forward.append(board[row + ((j + 1) * directions[i][0])][col + ((j + 1) * directions[i][1])])
                        except IndexError:
                            break
                    tiles[directions[i]] = forward
                adjacent_tiles[(row, col)] = tiles
            else:
                adjacent_tiles[(row, col)] = -1
            total_score = 0
            try:
                for id, values in adjacent_tiles.items():
                    directions = list(values.keys())
                    for i in range(0, len(directions), 2):  # Iterate in pairs (opposing directions)
                        dir1, dir2 = directions[i], directions[i + 1]
                        line1, line2 = values[dir1], values[dir2]
                        score1 = 0
                        score2 = 0
                        first = 0
                        # Convert line so that the first non-zero cell is 1 and any opposing non-zero number is 2
                        for j in range(len(line1)):
                            try:
                                if first == 0 and line1[j] > 0:
                                    first = line1[j]
                                if line1[j] > 0:
                                    if line1[j] == first:
                                        line1[j] = 1
                                    else:
                                        line1[j] = 2
                            except IndexError:
                                break
                        first = 0
                        for k in range(len(line2)):
                            try:
                                if first == 0 and line2[k] > 0:
                                    first = line2[k]
                                if line2[k] > 0:
                                    if line2[k] == first:
                                        line2[k] = 1
                                    else:
                                        line2[k] = 2
                            except IndexError:
                                break
                        lines = [str(line1), str(line2)]
                        for category in score_board:
                            for key in category.keys():
                                for item in category[key]:
                                    for l in range(len(lines)):
                                        if lines[l] in item:
                                            if l == 0:
                                                score1 += item[lines[l]]
                                            else:
                                                score2 += item[lines[l]]
                        if score1 > 0 and score2 > 0:
                            total_score += (score1 + score2)
                        else:
                            total_score += (score1 + score2)
            except AttributeError:
                total_score = -1
            scored_board[row][col] = total_score
    scores_normalized = []
    max_score = int(np.amax(scored_board))
    scored_board_flat = scored_board.flatten()
    # normalize the score for ai training purposes
    for i in range(len(scored_board_flat)):
        new_normalized_score = 0
        if max_score > 0:
            new_normalized_score = (scored_board_flat[i] / (max_score / 2) - 1)
        if new_normalized_score < 0:
            new_normalized_score = 0
        scores_normalized.append(new_normalized_score)
    return max_score, scored_board, scores_normalized


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


def convert_to_one_hot(board, player_id):
    board = np.array(board)
    height, width = board.shape
    one_hot_board = np.zeros((3, height, width), dtype=np.float32)
    one_hot_board[0] = (board == 0).astype(np.float32)
    if player_id == 1:
        one_hot_board[1] = (board == 1).astype(np.float32)  # AI's pieces as Player 1
        one_hot_board[2] = (board == 2).astype(np.float32)  # Enemy's pieces as Player 2
    else:
        one_hot_board[1] = (board == 2).astype(np.float32)  # AI's pieces as Player 2
        one_hot_board[2] = (board == 1).astype(np.float32)
    return one_hot_board


def run(instance, game_number, train, record_replay=False, moves:dict=None):
    # Main game loop
    global window_name, victory_text, current_player
    for p in players:
        if p.TYPE == "MM-AI":
            p.ai.model.load_model()
            p.ai.train = train
    # Initialize Pygame
    pygame.display.set_icon(pygame.image.load('res/ico.png'))
    pygame.init()
    pygame.display.set_caption(window_name)
    instance.winning_cells = []
    running = True
    if record_replay:
        p1_moves = []
        p2_moves = []
    if moves is not None:
        move_id = 0
        position = list(moves.keys())
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
                            if current_player == 1 and record_replay:
                                p1_moves.append((row, col))
                            elif record_replay:
                                p2_moves.append((row, col))
                            players[current_player - 1].moves += 1
                            if check_win(row, col, current_player, instance):
                                victory_text = f"Player {current_player} wins!"
                                running = False
                            else:
                                # Switch player if neither player have won
                                current_player = 3 - current_player
            # TestAI move
            elif players[current_player-1].TYPE == "AI" and not testai.check_game_over(instance):
                if instance.ai_delay:
                    time.sleep(random.uniform(0.25, 1.0))   # randomize ai "thinking" time
                ai_row, ai_col = testai.ai_move(instance, players[current_player-1].ID)
                testai.make_move((ai_row, ai_col), current_player, instance)
                players[current_player-1].moves += 1
                if current_player == 1 and record_replay:
                    p1_moves.append((ai_row, ai_col))
                elif record_replay:
                    p2_moves.append((ai_row, ai_col))
                if check_win(ai_row, ai_col, current_player, instance):
                    victory_text = f"AI {players[current_player-1].ID} wins!"
                    running = False
                else:
                    current_player = 3 - current_player
            # MM-AI move
            elif players[current_player-1].TYPE == "MM-AI":
                if instance.ai_delay:
                    time.sleep(random.uniform(0.25, 1.0))   # randomize AI "thinking" time
                one_hot_board = convert_to_one_hot(instance.board, players[current_player-1].ID)
                mm_ai = players[current_player-1].ai
                mm_ai.set_game(one_hot_board)
                old_state = instance.board
                max_score, scores, scores_normalized = calculate_score(instance.board)
                action = mm_ai.get_action(instance.board, one_hot_board, scores_normalized)
                np_scores = np.array(scores).reshape(15, 15)
                short_score = np_scores[action[0]][action[1]]
                if max_score <= 0:
                    # prevent division with negative values or zero
                    score = 0
                else:
                    score = short_score / max_score
                if current_player == 1 and record_replay:
                    p1_moves.append(action)
                elif record_replay:
                    p2_moves.append(action)
                players[current_player - 1].weighed_moves.append(score)
                instance.board[action[0]][action[1]] = current_player
                game_over = check_win(action[0], action[1], current_player, instance)
                next_max_score, next_scores, next_scores_normalized = calculate_score(instance.board)
                # Train the AI
                if train:
                    mm_ai.remember(old_state, action, score, instance.board, game_over)
                    mm_ai.train_short_memory(one_hot_board, action, short_score, scores, convert_to_one_hot(instance.board, players[current_player-1].ID), next_scores, game_over)
                    players[current_player - 1].move_loss.append(mm_ai.loss)
                players[current_player-1].final_action = action
                players[current_player - 1].moves += 1
                if game_over:
                    victory_text = f"MM-AI {players[current_player - 1].ID} wins!"
                    running = False
                else:
                    current_player = 3 - current_player
            #Replay
            elif players[current_player - 1].TYPE == "replay":
                if instance.ai_delay:
                    time.sleep(random.uniform(0.25, 1.0))   # randomize ai "thinking" time
                instance.board[position[move_id][0]][position[move_id][1]] = current_player
                if check_win(position[move_id][0], position[move_id][1], current_player, instance):
                    victory_text = f"AI {players[current_player-1].ID} wins!"
                    running = False
                else:
                    current_player = 3 - current_player
                    move_id += 1
            draw_board(instance)
            pygame.display.flip()
            window_name = "Gomoku - Game: " + str(game_number) + " - Player " + str(current_player)
            pygame.display.set_caption(window_name)
        else:
            victory_text = "TIE"
            current_player = -1
            running = False

    # End game
    stats.log_message(victory_text)
    pygame.display.set_caption("Gomoku - Game: " + str(game_number) + " - " + victory_text)
    update_player_stats(instance, current_player-1)
    if record_replay:
        filereader.save_replay(p1_moves, p2_moves)
    # For any MM-AI, train for long memory and save model
    data = {}
    loss_data = {}
    move_loss_data = {}
    for p in players:
        if p.TYPE == "MM-AI" and train:
            p.ai.remember(instance.board, p.final_action, p.score, instance.board, True)
            p.ai.train_long_memory()
            p.score_loss.append(p.ai.loss)
            move_loss = [float(val) for val in p.move_loss]
            p.final_move_loss.append(sum(move_loss)/len(move_loss))
            p.ai.model.save_model()
            p.final_move_scores.append(sum(p.weighed_moves)/len(p.weighed_moves))
            stats.log_message(f"{p.TYPE} {p.ID}: score loss: {float(p.ai.loss)}")
            stats.log_message(f"{p.TYPE} {p.ID}: move loss: {sum(p.move_loss)/len(p.move_loss)}")
        p.reset_score()
        if instance.last_round:
            if p.TYPE == "MM-AI" and train:
                data[f"{p.TYPE} {p.ID}: game accuracy"] = p.weighed_scores
                data[f"{p.TYPE} {p.ID}: move accuracy"] = p.final_move_scores
                loss_data[f"{p.TYPE} {p.ID}: score loss"] = [float(val) for val in p.score_loss]
                move_loss_data[f"{p.TYPE} {p.ID}: move loss"] = p.final_move_loss
                stats.log_message(f"{p.TYPE} {p.ID}: average score loss: {sum([float(val) for val in p.score_loss]) / len([float(val) for val in p.score_loss])}")
                stats.log_message(f"{p.TYPE} {p.ID}: average move loss: {sum(p.final_move_loss) / len(p.final_move_loss)}")
            p.reset_all_stats()
    if len(data) > 0:
        stats.plot_graph(data, 'accuracy')
    if len(loss_data) > 0:
        stats.plot_graph(loss_data, 'loss data')
    if len(move_loss_data) > 0:
        stats.plot_graph(move_loss_data, 'loss data')
    time.sleep(instance.SLEEP_BEFORE_END)
    reset_game(instance)


pygame.quit()
