import keras.optimizers
import keras.layers.core
import keras.models
import torch
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class GomokuAI:
    def build_model(self, input_dim: int, output: int):
        model = Sequential()
        model.add(Dense(self.board_size**2, activation='relu', input_dim=input_dim))
        # model.add(Dense(50;300;50, activation='relu', input_dim=input_dim))
        model.add(Dense((self.board_size**2)*2, activation='relu'))
        model.add(Dense(self.board_size**2, activation='relu'))
        model.add(Dense(output, activation='softmax'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model




    def __init__(self, _board_size = 15):
        self.n_games = 0
        self.learning_rate = 0.0005
        self.len_action = len(self.Action().to_array())
        self.len_state = len(self.State().to_array())
        self.board_size = _board_size
        self.gamma = 0.9
        self.epsilon = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(self.board_size**2, (self.board_size**2)*2, 1)
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)
        #self.model = self._build_model2(self.len_state, self.len_action)

    def get_state(self, game):
        pass

    def remember(self):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_action(self, state):
        pass


def train():
    plot_score = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    ai = GomokuAI()
    while True:
        state_old = ai.get_state(game)


if __name__ == '__main__':
    train()