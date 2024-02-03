import keras.optimizers
import keras.layers.core
import keras.models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class ConvNet(nn.Module):
    def __init__(self, input_dim):
        super(ConvNet, self).__init__()
        # Define your CNN architecture here
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GomokuAI:
    def __init__(self, _game, _board_size = 15):
        self.n_games = 0
        self.learning_rate = 0.0005
        self.len_action = len(self.Action().to_array())
        self.len_state = len(self.State().to_array())
        self.board_size = _board_size
        self.gamma = 0.9
        self.epsilon = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        #self.model = Linear_QNet(self.board_size**2, (self.board_size**2)*2, 1)
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)
        self.model = self._build_model(self.board_size)
        self.game = _game

    def build_model(self, input_dim: int) -> ConvNet:
        return ConvNet(input_dim)

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

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        # Predict Q-values for current states
        q_pred = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Predict Q-values for next states
        q_next = self.model(next_states).detach()
        q_target = rewards + self.gamma * torch.max(q_next, dim=1)[0] * (~torch.tensor(dones))

        # Loss and backpropagation
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        # Convert state, next_state to tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        # Convert action to tensor, reward to tensor
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)

        # Predict Q-value for the current state
        q_pred = self.model(state).gather(1, action)

        # Calculate target Q-value
        q_next = self.model(next_state).detach()
        q_target = reward + self.gamma * torch.max(q_next).unsqueeze(1)

        # Loss and backpropagation
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        # Exploration vs Exploitation: Epsilon-Greedy Strategy
        if random.random() < self.epsilon:
            # Exploration: Random Move
            action = self.get_random_action()
        else:
            # Exploitation: Best Move According to the Model
            current_state = self.get_state(state)
            prediction = self.model(current_state)
            action = torch.argmax(prediction).item()  # Convert to actual move/action

        # Decay Epsilon Over Time
        if self.epsilon > 0.01:  # Setting a lower bound on epsilon
            self.epsilon *= 0.99  # Adjust decay rate as needed

        return action

    def get_random_action(self):
        while True:
            p = random.randint(0, len(self.game))
            if self.game[p] == 0:
                return p

    def ai_move(self):
        board_state = self.get_state(self.game)


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