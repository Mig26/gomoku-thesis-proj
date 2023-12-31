import random
DEPTH = 5


def check_line(row, col, direction, board):
    score_own = score_enemy = previous = 0
    multiplier = 1
    adjacency_loss = 1
    for i in range(DEPTH-1):
        try:
            if board[row + (direction[0] * (i + 1))][col + (direction[1] * (i + 1))] != 0:
                if previous == board[row + (direction[0] * (i+1))][col + (direction[1] * (i+1))]:
                    multiplier *= (1 + multiplier)
                else:
                    multiplier = 1
                    score_own = 0
                    score_enemy = 0
                if board[row + (direction[0] * (i+1))][col + (direction[1] * (i+1))] == 1:
                    score_enemy += 2 * multiplier - i
                elif board[row + (direction[0] * (i+1))][col + (direction[1] * (i+1))] == 2:
                    score_own += 3.5 * multiplier - i
                previous = board[row + (direction[0] * (i + 1))][col + (direction[1] * (i + 1))]
            elif i == 0:
                adjacency_loss = 8
            try:
                if board[row + direction[0]][col + direction[1]] == board[row - direction[0]][col - direction[1]]:
                    if board[row + (direction[0])][col + direction[1]] == 1 and score_enemy > 10:
                        score_enemy *= 4
                    elif score_own > 10:
                        score_own *= 4
            except IndexError:
                pass
        except IndexError:
            break
    score_own = score_own / adjacency_loss
    score_enemy = score_enemy / adjacency_loss
    return int(score_own), int(score_enemy)


def evaluate_board(instance):
    scores = {}
    board = instance.board
    grid_size = instance.GRID_SIZE
    directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, 1), (-1, -1)]
    for row in range(grid_size):
        for col in range(grid_size):
            if board[row][col] != 0:
                pass
            else:
                score_own = score_enemy = 0
                try:
                    for i in range(len(directions)):
                        score_own_, score_enemy_ = check_line(row, col, directions[i], board)
                        score_own += score_own_
                        score_enemy += score_enemy_
                except IndexError:
                    pass
                score = 0
                if score_own > score_enemy:
                    score = score_own
                else:
                    score = score_enemy
                # score = score_enemy - score_own
                if score < 0:
                    score = 0
                scores[(row, col)] = score
    return scores


def make_move(move, player, instance):
    row, col = move
    instance.board[row][col] = player


def get_available_moves(instance):
    moves = []
    for row in range(instance.GRID_SIZE):
        for col in range(instance.GRID_SIZE):
             if instance.board[row][col] == 0:
                 moves.append((row, col))
    return moves


def check_game_over(instance):
    for row in range(instance.GRID_SIZE):
        for col in range(instance.GRID_SIZE):
            if instance.board[row][col] == 0:
                return False
    return True


def ai_move(instance):
    moves = get_available_moves(instance)
    scores = evaluate_board(instance)
    max_score = max(scores.values())
    try:
        best_move = random.choice([k for k,v in scores.items() if v == max_score])
    except IndexError:
        best_move = random.choice(moves)
    # print("Max score:", max_score, ", best move:", best_move)
    return best_move
