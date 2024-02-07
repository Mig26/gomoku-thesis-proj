import os.path

import torch
import torch.nn as nn
import torch.optim as optim
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
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(output_dim, 1)
        # self.fc2 = nn.Linear(256, 1)
        # self.fc1 = nn.Linear(board_size * board_size * 128, 1024)
        # self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten the tensor
        # x = torch.relu(self.fc1(x))
        x = self.fc1(x)
        return x

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
    def __init__(self, _board_size = 15):
        self.n_games = 0
        self.game = None
        self.learning_rate = 0.00025
        # self.len_action = len(self.Action().to_array())
        # self.len_state = len(self.State().to_array())
        self.board_size = _board_size
        self.gamma = 0.2
        self.epsilon = 0.25
        self.memory = deque(maxlen=MAX_MEMORY)
        #self.model = Linear_QNet(self.board_size**2, (self.board_size**2)*2, 1)
        self.model = self.build_model(self.board_size**2)
        self.model.load_model()
        self.model.eval()
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def set_game(self, _game):
        self.game = _game

    def build_model(self, input_dim: int) -> ConvNet:
        return ConvNet(input_dim, input_dim*2, 128)

    def get_state(self, game):
        return torch.tensor(game, dtype=torch.float32).unsqueeze(0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            mini_batch = self.memory
        else:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        state = torch.tensor(mini_batch[0][0], dtype=torch.float).view(1, -1).unsqueeze(-1).unsqueeze(-1)
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float)#.mean(dim=1, keepdim=True)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        # Predict Q-values for current states
        # q_pred = self.model(state)#.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # q_pred = self.model(state.squeeze(0).squeeze(0).view(1, -1).unsqueeze(-1).unsqueeze(-1))
        q_pred = self.model(state)

        # Predict Q-values for next states
        q_next = next_states.detach()
        q_next_max = torch.max(q_next, dim=1)[0].unbind(dim=1)[0]
        q_target = rewards + (self.gamma * q_next_max * (~torch.tensor(dones)))

        # Loss and backpropagation
        loss = self.criterion(q_pred, q_target.unsqueeze(1))
        print(f"loss: {loss}")
        loss.requires_grad_(requires_grad=True)
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        state_flattened = state.view(-1)    # flattened state
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        action_index = action[0, 0] * state.size(2) + action[0, 1]  # row * number_of_columns + column
        action_index = action_index.long()

        q_pred = self.model(state.squeeze(0).squeeze(0).view(1, -1).unsqueeze(-1).unsqueeze(-1))#.gather(0, action.unsqueeze(0).unsqueeze(-1))
        q_next = next_state
        q_target = reward
        if not done:
            q_target = reward + (self.gamma * torch.max(q_next))

        # loss = self.criterion(target, pred)
        loss = self.criterion(q_pred, q_target.unsqueeze(1))
        loss.requires_grad_(requires_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

    def get_action(self, state):
        valid_moves = self.get_valid_moves(state)
        # Exploration vs Exploitation: Epsilon-Greedy Strategy
        if random.random() < self.epsilon:
            # Exploration: Random Move
            action = self.id_to_move(self.get_random_action(), valid_moves)
        else:
            # Exploitation: Best Move According to the Model
            # current_state = self.get_state(state)
            # current_state = torch.tensor(self.get_state(state), dtype=torch.float)
            current_state = self.get_state(state).clone().detach()
            # prediction = self.model(current_state)
            prediction = current_state
            action = self.id_to_move(torch.argmax(prediction).item(), valid_moves)
            if action is None:
                # if no action, switch to exploration
                action = self.id_to_move(self.get_random_action(), valid_moves)

        # Decay Epsilon Over Time
        if self.epsilon > 0.01:  # Setting a lower bound on epsilon
            self.epsilon *= 0.99  # Adjust decay rate as needed
        return action

    def get_random_action(self):
        while True:
            p = random.randint(0, len(self.game)-1)
            if self.game[p][random.randint(0, len(self.game[p])-1)] == 0:
                break
        return p

    def calculate_short_score(self, move: tuple, board: tuple):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, 1), (-1, -1)]
        score = 0
        try:
            for i in range(len(directions)):
                current_score = 0
                for j in range(3):
                    if j == 0:
                        first_spot = board[move[0]+((j+1)*directions[i][0])][move[1]+((j+1)*directions[i][1])]
                    current_spot = board[move[0]+((j+1)*directions[i][0])][move[1]+((j+1)*directions[i][1])]
                    if current_spot == 1 or current_spot == 2:  # run only if current spot is not empty
                        if j > 0:
                            previous_spot = board[move[0] + ((j - 1) * directions[i][0])][
                                move[1] + ((j - 1) * directions[i][1])]
                            if current_spot == previous_spot:
                                current_score += j+1    # increase score if the current and previous spots are of the same color
                            elif current_spot != previous_spot and current_spot != 0:   # a situation where a line is blocked
                                for k in range(3):  # check opposing direction for continuation
                                    opposing_spot = board[move[0]-((j+1)*directions[i][0])][move[1]-((j+1)*directions[i][1])]
                                    if opposing_spot != 0 and opposing_spot == first_spot:
                                        current_score += k+1    # increase score if opposing direction has the same color as the first spot of the current direction
                                    elif opposing_spot != 0 and opposing_spot != first_spot:
                                        if (j+1)+(k+1) <= 4: # if the lines are too short and blocked from both sides, don't reward
                                            current_score = 0
                                        break
                                    else:
                                        break
                                break
                        elif current_spot != 0: # this condition only applies if j == 0
                            current_score += j+1
                    else:
                        break   # exit immediately if hits an empty spot
                score += current_score
        except IndexError:
            pass
        return score

    def calculate_short_max_score(self, board: tuple, board_size = 15):
        moves = []
        for row in range(board_size):
            for col in range(board_size):
                moves.append(self.calculate_short_score((row, col), board))
        return max(moves)

    def train(self):    # Siirrä tämä koodi gomoku.py
        plot_score = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        ai = GomokuAI()
        #while True:
        #    state_old = ai.get_state(game)

        for game_number in range(NUMBER_OF_GAMES):
            state = game.reset()  # Reset the game to the initial state

            while not game.is_finished():
                action = ai.get_action(state)
                next_state, reward, done = game.play(action)  # Play the action and get feedback
                ai.remember(state, action, reward, next_state, done)  # Store in memory
                ai.train_short_memory(state, action, reward, next_state, done)  # Train short-term memory

                state = next_state

                if done:
                    # Game is finished
                    ai.train_long_memory()  # Train long-term memory
                    break

            # Adjust epsilon after each game
            if ai.epsilon > MIN_EPSILON:
                ai.epsilon *= EPSILON_DECAY_RATE

            # Log results, monitor performance, etc.
            # ...

if __name__ == '__main__':
    train()