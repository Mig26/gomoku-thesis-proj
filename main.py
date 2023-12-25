import gomoku
import filereader


def main():
    game_instance = gomoku.GomokuGame(filereader.create_gomoku_game("consts.json"))
    gomoku.run(game_instance)


if __name__ == '__main__':
    main()
