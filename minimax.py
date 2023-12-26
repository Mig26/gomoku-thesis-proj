import random
DEPTH = 32


# Function to evaluate the board for the Minimax algorithm
def evaluate_board(instance):
    scores = []
    board = instance.board
    grid_size = instance.GRID_SIZE
    for row in range(grid_size):
        for col in range(grid_size):
            score = 0
            if board[row][col] != 0:
                pass
            else:
                try:
                    if board[row-1][col] != 0:
                        if board[row-1][col] == 2:
                            score += 1
                        score += 1
                except IndexError:
                    pass
                try:
                    if board[row-1][col-1] != 0:
                        if board[row - 1][col - 1] == 2:
                            score += 1
                        score += 1
                except IndexError:
                    pass
                try:
                    if board[row-1][col+1] != 0:
                        if board[row - 1][col + 1] == 2:
                            score += 1
                        score += 1
                except IndexError:
                    pass
                try:
                    if board[row][col-1] != 0:
                        if board[row][col-1] == 2:
                            score += 1
                        score += 1
                except IndexError:
                    pass
                try:
                    if board[row+1][col] != 0:
                        if board[row+1][col] == 2:
                            score += 1
                        score += 1
                except IndexError:
                    pass
                try:
                    if board[row+1][col+1] != 0:
                        if board[row+1][col+1] == 2:
                            score += 1
                        score += 1
                except IndexError:
                    pass
                try:
                    if board[row+1][col-1] != 0:
                        if board[row+1][col-1] == 2:
                            score += 1
                        score += 1
                except IndexError:
                    pass
                try:
                    if board[row][col+1] != 0:
                        if board[row][col + 1] == 2:
                            score += 1
                        score += 1
                except IndexError:
                    pass
            scores.append(score)
    return scores


# Function to make a move
def make_move(move, player, instance):
    row, col = move
    instance.board[row][col] = player


# Function to undo a move
def undo_move(move, instance):
    row, col = move
    instance.board[row][col] = 0


def maximin(depth, instance):
    if depth >= 0:
        moves = get_available_moves(instance)
        scores = minimin(depth - 1, instance)
        max_score = max(scores)
        try:
            move = moves[random.choice([i for i in range(len(scores)) if scores[i] == max_score])]
            make_move(move, 2, instance)
            eval = evaluate_board(instance) + scores
            undo_move(move, instance)
            return eval
        except IndexError:
            return evaluate_board(instance)
    return evaluate_board(instance)


def minimin(depth, instance):
    if depth >= 0:
        moves = get_available_moves(instance)
        scores = maximin(depth - 1, instance)
        max_score = max(scores)
        try:
            move = moves[random.choice([i for i in range(len(scores)) if scores[i] == max_score])]
            make_move(move, 1, instance)
            eval = evaluate_board(instance) + scores
            undo_move(move, instance)
            return eval
        except IndexError:
            return evaluate_board(instance)
    return evaluate_board(instance)


# Minimax algorithm for AI player
def minimax(depth, maximizing_player, instance):
    if depth == 0 or check_game_over(instance):
        return evaluate_board(instance)
    if maximizing_player:
        return maximin(DEPTH, instance)
    else:
        return minimin(DEPTH, instance)


# Function to get available moves
def get_available_moves(instance):
    moves = []
    for row in range(instance.GRID_SIZE):
        for col in range(instance.GRID_SIZE):
             if instance.board[row][col] == 0:
                 moves.append((row, col))
    return moves


# Function to check if the game is over
def check_game_over(instance):
    for row in range(instance.GRID_SIZE):
        for col in range(instance.GRID_SIZE):
            if instance.board[row][col] == 0:
                return False
    return True


# Function for AI player's move
def ai_move(instance):
    moves = get_available_moves(instance)
    scores = minimax(DEPTH, False, instance)
    max_score = max(scores)
    try:
        best_move = moves[random.choice([i for i in range(len(scores)) if scores[i] == max_score])]
    except IndexError:
        best_move = random.choice(moves)
    print("Max score:", max_score, ", best move:", best_move)
    return best_move
