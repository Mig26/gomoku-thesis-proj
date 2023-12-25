DEPTH = 2


# Function to evaluate the board for the Minimax algorithm
def evaluate_board(instance):
    score = 0
    for row in range(instance.GRID_SIZE):
        for col in range(instance.GRID_SIZE):
            if instance.board[row][col] == 1:
                score += 1
            elif instance.board[row][col] == 2:
                score -= 1
    if score > 0:
        print("Grid score:", score)
    return score


# Minimax algorithm for AI player
def minimax(depth, maximizing_player, instance):
    if depth == 0 or check_game_over(instance):
        return evaluate_board(instance)

    if maximizing_player:
        max_eval = float('-inf')
        for move in get_available_moves(instance):
            make_move(move, 2, instance)
            eval = minimax(depth - 1, False, instance)
            undo_move(move, instance)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_available_moves(instance):
            make_move(move, 1, instance)
            eval = minimax(depth - 1, True, instance)
            undo_move(move, instance)
            min_eval = min(min_eval, eval)
        return min_eval


# Function to get available moves
def get_available_moves(instance):
    moves = []
    for row in range(instance.GRID_SIZE):
        for col in range(instance.GRID_SIZE):
            if instance.board[row][col] == 0:
                moves.append((row, col))
    return moves


# Function to make a move
def make_move(move, player, instance):
    row, col = move
    instance.board[row][col] = player


# Function to undo a move
def undo_move(move, instance):
    row, col = move
    instance.board[row][col] = 0


# Function to check if the game is over
def check_game_over(instance):
    for row in range(instance.GRID_SIZE):
        for col in range(instance.GRID_SIZE):
            if instance.board[row][col] == 0:
                return False
    return True


# Function for AI player's move
def ai_move(instance):
    best_score = float('-inf')
    best_move = None
    for move in get_available_moves(instance):
        make_move(move, 2, instance)
        score = minimax(DEPTH, False, instance)
        undo_move(move, instance)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move