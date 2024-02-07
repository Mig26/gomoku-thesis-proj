import logging
import math
import numpy as np
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt

should_log = True


def setup_logging(player0_type, player1_type):
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_filename = os.path.join(log_folder, f"logs-{player0_type}-{player1_type}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt")

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    if should_log:
        log_message("Log file created.")
    else:
        log_message("Log file NOT created.")


def log_message(message):
    if should_log:
        logging.info(message)
    print(message)


def log_win(players):
    for i in range(len(players)):
        log = f"{players[i].TYPE} {players[i].ID} - score: {players[i].score} - moves: {players[i].moves} - wins: {players[i].wins} - losses: {players[i].losses}"
        print(log)
        if should_log:
            logging.info(log)


def plot_graph(data: dict, data_name='data', title='title'):
    plt.figure(figsize=(10, 6))
    for k, v in data.items():
        plt.plot(v, 'o', label=k)
        running_avg = np.cumsum(v) / (np.arange(len(v)) + 1)
        plt.plot(running_avg, label=f'{k} avg')
        plt.ylabel(data_name)
        plt.xlabel('round')
    plt.legend()
    plt.title(title)
    plt.show()
