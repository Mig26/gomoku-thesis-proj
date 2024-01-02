import logging
import datetime
import os


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
    log_message("Log file created.")


def log_message(message):
    logging.info(message)
    print(message)


def log_win(players):
    for i in range(len(players)):
        log = f"{players[i].TYPE} {players[i].ID} - score: {players[i].score} - moves: {players[i].moves} - wins: {players[i].wins} - losses: {players[i].losses}"
        print(log)
        logging.info(log)

