import os.path
from math import isclose
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 1_000_000
BATCH_SIZE = 10000
MIN_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.999


class ConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConvNet, self).__init__()
        # Define your CNN architecture here
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, hidden_dim, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=5, stride=1),
            torch.nn.Dropout(p=0.05))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim * input_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=5, stride=1),
            torch.nn.Dropout(p=0.1))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, output_dim, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=5),
            torch.nn.Dropout(p=0.2))
        self.fc2 = torch.nn.Linear(input_dim * input_dim, input_dim * input_dim, bias=True)
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=input_dim * input_dim, out_channels=input_dim * input_dim, kernel_size=5, padding=2)
        # self.fc1 = nn.Linear(output_dim, 1)
        self.fc3 = nn.Linear(output_dim, input_dim, bias=False)

        # self.fc1 = nn.Conv2d(in_channels=output_dim, out_channels=2, kernel_size=3, padding=1)
        # self.fc2 = nn.Linear(256, 1)
        self.fc1 = nn.Linear(output_dim, input_dim * input_dim)
        # self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        # print(f"x in:\n{x}")
        out = F.relu(self.layer1(x))
        # print(f"out after first conv:\n{out}")
        # out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        # print(f"out after second conv:\n{np.shape(out)}")
        # out = out.squeeze(0).squeeze(-1).squeeze(-1)
        # print(np.shape(out))
        # out = out.mean(dim=[1, 2], keepdim=True).squeeze(-1).squeeze(-1)
        # print(f"out after squeeze and mean:\n{out}")
        # out = self.fc1(out)
        # print(f"out after first fully connected:\n{out}")
        # out = self.fc2(out)
        # print(f"out after second fully connected:\n{out}")
        # print(f"max out: {torch.max(out)}")
        return out
    # x = F.relu_(self.conv1(x))
    # x = F.relu_(self.conv2(x))
    # x = x.squeeze(0)
    # x = x.mean(dim=[1, 2], keepdim=True).squeeze(-1).squeeze(-1)
    # x = self.fc1(x)
    # return x

    def load_model(self, file_name='model.pth'):
        model_folder = './data/'
        full_path = os.path.join(model_folder, file_name)
        if os.path.isfile(full_path):
            print("A model exist, loading model.")
            self.load_state_dict(torch.load(full_path))
        else:
            print("No model exist. Creating a new model.")

    def save_model(self, file_name='model.pth'):
        model_folder = './data/'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        full_path = os.path.join(model_folder, file_name)
        torch.save(self.state_dict(), full_path)
        print(f"Model saved to directory {full_path}.")


class GomokuAI:
    def __init__(self, _board_size=15):
        self.n_games = 0
        self.game = None
        self.learning_rate = 0.00075
        # self.learning_rate = 0.00000000025
        # self.len_action = len(self.Action().to_array())
        # self.len_state = len(self.State().to_array())
        self.board_size = _board_size
        self.gamma = 0.2
        self.epsilon = 0.25
        self.memory = deque(maxlen=MAX_MEMORY)
        # self.model = Linear_QNet(self.board_size**2, (self.board_size**2)*2, 1)
        self.model = self.build_model(self.board_size)
        # self.model.load_model()
        # self.model.eval()
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()
        self.loss = 0

    def set_game(self, _game):
        self.game = _game

    def build_model(self, input_dim: int) -> ConvNet:
        return ConvNet(input_dim, 30, 255)

    def get_state(self, game):
        return torch.tensor(game, dtype=torch.float32).unsqueeze(0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def adjust_epsilon(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY_RATE

    def train_long_memory(self):
        self.model.train()
        if len(self.memory) < BATCH_SIZE:
            mini_batch = self.memory
        else:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        state = torch.tensor(mini_batch[0][0], dtype=torch.float).view(1, -1).unsqueeze(-1).unsqueeze(-1)
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float).unsqueeze(0)  # .unsqueeze(0)
        states = torch.transpose(states, 0, 1)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        # Predict Q-values for current states
        q_pred = rewards  # .gather(1, actions.unsqueeze(-1).unsqueeze(0)).squeeze(-1)
        # print(f"final q_pred: {q_pred}")
        # q_pred = self.model(state.squeeze(0).squeeze(0).view(1, -1).unsqueeze(-1).unsqueeze(-1))
        # q_pred = self.model(state)

        # Predict Q-values for next states
        q_next = next_states.detach()
        q_next_max = torch.max(q_next, dim=1)[0].unbind(dim=1)
        # q_target = rewards + (self.gamma * q_next_max * (~torch.tensor(dones)))
        q_target = rewards + (self.gamma * (~torch.tensor(dones)))
        # Loss and backpropagation
        loss = self.criterion(q_pred, q_target)
        # loss = self.criterion(q_pred, q_target.unsqueeze(1))
        loss.requires_grad_(requires_grad=True)
        loss.backward()
        self.loss = loss.detach().numpy()  # for logging purposes
        self.optimizer.step()
        self.adjust_epsilon()
        # self.optimizer.zero_grad(set_to_none=False)

    def train_short_memory(self, state, action, reward, scores, next_state, next_scores, done):
        self.model.train()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        np_scores = np.array(scores).reshape(15, 15)
        np_next_scores = np.array(next_scores).reshape(15, 15)
        scores_tensor = torch.tensor(np_scores, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        next_scores_tensor = torch.tensor(np_next_scores, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # state_score = torch.cat((scores_tensor, state), dim=1)
        state_flattened = state.view(-1)  # flattened state
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        # next_state_score = torch.cat((next_scores_tensor, next_state), dim=1)
        action = torch.tensor([action], dtype=torch.long)
        # print(f"scores:\n{reward}")
        reward = torch.tensor([reward], dtype=torch.float)

        # self.model.eval()
        # action_index = action[0, 0] * state.size(2) + action[0, 1]  # row * number_of_columns + column
        # action_index = action_index.long()
        # q_pred = self.model(state_flattened.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))#.view(1, -1).unsqueeze(-1).unsqueeze(-1))#.squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)).squeeze(0).squeeze(0)#.gather(0, action).view(1, -1).unsqueeze(-1).unsqueeze(-1)  #
        # q_pred = self.model(state.squeeze(0).squeeze(0).view(1, -1).unsqueeze(-1).unsqueeze(-1))#.gather(-2, action.unsqueeze(0).unsqueeze(-1))
        q_pred = self.model(state)
        # print(f"next state: {np.shape(next_state_score)}")
        q_next = self.model(next_state)
        # print(f"q next: {q_next}")
        q_target = q_pred.clone()
        q_new = q_pred + self.learning_rate * (reward + self.gamma * torch.argmax(q_pred - q_next))

        # print(f"torch max: {(reward + self.gamma * torch.max(q_pred))}")
        # print(f"q new: {q_new}")
        # q_new = reward + self.gamma * torch.max(self.model(next_state.view(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)))
        # if not done:
        #     q_target = reward + (self.gamma * torch.max(q_next))
        # q_target[torch.argmax(action).item()] = q_new
        loss = self.criterion(q_pred, q_new)
        # loss = self.criterion(q_pred, q_target.unsqueeze(1))
        loss.requires_grad_(requires_grad=True)
        loss.backward()
        self.loss = loss.detach().numpy()  # for logging purposes
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=False)

    def get_valid_moves(self, board):
        valid_moves = []
        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == 0:
                    valid_moves.append((row, col))
        return valid_moves

    def id_to_move(self, move_id, valid_moves):
        if move_id < len(valid_moves):
            return valid_moves[move_id]
        else:
            return None

    def get_action(self, state, one_hot_board, scores):
        valid_moves = self.get_valid_moves(state)
        np_scores = np.array(scores).reshape(15, 15)
        # np_scores = np.array(scores).flatten()
        scores_tensor = torch.tensor(np_scores, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        current_state = torch.tensor(self.get_state(one_hot_board), dtype=torch.float)
        # current_state = torch.tensor(self.get_state(state), dtype=torch.float)
        # current_state = torch.cat((scores_tensor, current_state), dim=1)
        # current_state = current_state.mean(dim=[1, 2], keepdim=True)
        action = None
        with torch.no_grad():
            prediction = self.model(current_state)
            # prediction = self.model(scores_tensor)
            # prediction = current_state
            # prediction = self.model(current_state.view(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        # Exploration vs Exploitation: Epsilon-Greedy Strategy
        if random.random() < self.epsilon:
            # Exploration: Random Move
            print("Exploration")
            num_moves_to_select = max(int(len(valid_moves) * .025), 1)
            if num_moves_to_select > 0:
                try:
                    top_moves_indices = torch.topk(prediction.flatten(), k=num_moves_to_select-1).indices
                    action = self.id_to_move(top_moves_indices[torch.randint(len(top_moves_indices), (1,))].item(), valid_moves)
                except RuntimeError:
                    action = None
            # action = self.id_to_move(self.get_random_action(state), valid_moves)
        else:
            # Exploitation: Best Move According to the Model
            print("Exploitation")

            state_concat = torch.cat((self.get_state(one_hot_board), scores_tensor), dim=1)
            # current_state = self.get_state(state_concat)

            # current_state = self.get_state(state_concat).clone().detach()

            # num_moves_to_select = max(int(len(valid_moves) * .025), 1)
            # if num_moves_to_select > 0:
            #     top_moves_indices = torch.topk(prediction.flatten(), k=num_moves_to_select).indices
            #     action = self.id_to_move(top_moves_indices[torch.randint(len(top_moves_indices), (1,))].item(), valid_moves)
            # else:
            # prediction = self.model(prediction)
            # pred_possible_moves = round(float(torch.max(prediction)), 4)
            pred_possible_moves = int(torch.max(prediction))
            pred_indices = np.where(np_scores == pred_possible_moves)
            # pred_indices = np.where(np.around(prediction, 4) == pred_possible_moves)
            if len(pred_indices[0]) > 0:
                idx = random.randint(0, len(pred_indices[0])-1)
                # print(f"idx: {idx}, indices: {pred_indices} = {int(pred_indices[0][idx])}")
                # pred_indices = self.id_to_move(int(pred_indices[0][idx]), valid_moves)
                # if pred_indices is not None:
                    # action = (pred_indices[0], pred_indices[1])
                action = (pred_indices[0][idx], pred_indices[1][idx])
            else:
                action = None
            # action = self.id_to_move(torch.argmax(prediction).item(), valid_moves)
        while action is None:
            # if no action, switch to exploration
            print("Exploration")
            action = self.id_to_move(self.get_random_action(state), valid_moves)
        # Decay Epsilon Over Time
        self.adjust_epsilon()
        return action

    def get_random_action(self, board):
        while True:
            row = random.randint(0, len(board) - 1)
            col = random.randint(0, len(board) - 1)
            if board[row][col] == 0:
                p = (row % (len(board) - 1) * len(board)) + (col + 1)
                break
        return p

    def convert_to_one_hot(self, board, player_id):
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

    def calculate_short_score(self, move: tuple, board: tuple, max_score_calculation=False):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, 1), (-1, -1)]
        score = 0
        first_spot = None
        try:
            for i in range(len(directions)):
                current_score = 0
                for j in range(5):
                    current_spot = board[move[0] + ((j + 1) * directions[i][0])][move[1] + ((j + 1) * directions[i][1])]
                    if j == 0:
                        first_spot = current_spot
                    if not max_score_calculation:
                        # if current_spot == 1 or current_spot == 2:  # run only if current spot is not empty
                        if current_spot > 0:
                            print(
                                f"current spot: {current_spot}. Location: ({move[0] + ((j + 1) * directions[i][0])}, {move[1] + ((j + 1) * directions[i][1])})")
                            if first_spot is not None:
                                current_score += self.calculate_score(current_score, move, board, current_spot, j,
                                                                      directions[i], first_spot)
                            else:
                                current_score += self.calculate_score(current_score, move, board, current_spot, j,
                                                                      directions[i])
                            # previous_spot = board[move[0] + ((j - 1) * directions[i][0])][
                            #     move[1] + ((j - 1) * directions[i][1])]
                            # if current_spot == previous_spot:
                            #     current_score *= j+1    # increase score if the current and previous spots are of the same color
                            # elif current_spot != previous_spot and current_spot != 0:   # a situation where a line is blocked
                            #     for k in range(3):  # check opposing direction for continuation
                            #         opposing_spot = board[move[0]-((j+1)*directions[i][0])][move[1]-((j+1)*directions[i][1])]
                            #         if opposing_spot != 0 and opposing_spot == first_spot:
                            #             current_score += k+1    # increase score if opposing direction has the same color as the first spot of the current direction
                            #         elif opposing_spot != 0 and opposing_spot != first_spot:
                            #             if (j+1)+(k+1) <= 4: # if the lines are too short and blocked from both sides, don't reward
                            #                 current_score = 0
                            #             break
                            #         else:
                            #             break
                            #     break
                        # elif current_spot != 0: # this condition only applies if j == 0
                        #    current_score += j+1
                        else:
                            pass
                    elif max_score_calculation:
                        if board[move[0]][move[1]] != 0:
                            current_score = -1
                            break
                        else:
                            if first_spot != None:
                                current_score = self.calculate_score(current_score, move, board, current_spot, j,
                                                                     directions[i], first_spot)
                            else:
                                current_score = self.calculate_score(current_score, move, board, current_spot, j,
                                                                     directions[i])
                    else:
                        pass  # exit immediately if hits an empty spot and not max score count
                score += current_score
        except IndexError:
            pass
        if max_score_calculation and score < -1:
            score = -1  # clamp max score calculation min value to -1, which represents a non-valid move
        return score

    def calculate_score(self, current_score, move, board, current_spot, j, direction: tuple, first_spot=-1) -> int:
        score = 0
        previous_spot = board[move[0] + ((j - 1) * direction[0])][move[1] + ((j - 1) * direction[1])]
        if current_spot == previous_spot:
            if first_spot > 0:
                if current_score > 0:
                    score = current_score * (
                                j + 1)  # increase score if the current and previous spots are of the same color
                else:
                    score = j + 1
        else:
            if first_spot > 0:  # a situation where a line is blocked
                for k in range(3):  # check opposing direction for continuation
                    opposing_spot = board[move[0] - ((j + 1) * direction[0])][move[1] - ((j + 1) * direction[1])]
                    if opposing_spot != 0 and opposing_spot == first_spot:
                        score = current_score * (
                                    k + 1)  # increase score if opposing direction has the same color as the first spot of the current direction
                    elif opposing_spot != 0 and opposing_spot != first_spot:
                        if (j + 1) + (
                                k + 1) <= 4:  # if the lines are too short and blocked from both sides, don't reward
                            score = 0
            else:
                pass
        return score

    def calculate_short_max_score(self, board: tuple, board_size=15):
        moves = []
        for row in range(board_size):
            for col in range(board_size):
                score = self.calculate_short_score((row, col), board, True)
                moves.append(score)
                if score > 0:
                    print(f"score: {score}, location: {row}, {col}")
        moves_normalized = []
        for i in range(len(moves)):
            if max(moves) > 0:
                new_normalized_move = (moves[i] / (max(moves) / 2) - 1)
            else:
                new_normalized_move = 0
            if new_normalized_move < 0:
                new_normalized_move = 0
            moves_normalized.append(new_normalized_move)
        print(f"best score: {max(moves)}")
        np_moves_norm = np.array(moves)
        reshaped = np.reshape(np_moves_norm, (board_size, board_size))
        return max(moves), moves, moves_normalized
