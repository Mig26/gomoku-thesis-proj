import json


def create_gomoku_game(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        gomoku_config = data.get("gomoku", [])[0]
        values = []
        for key, value in gomoku_config.items():
            if type(value) is str:
                value = string_to_color(value)
            values.append(value)
        return values


def load(filename):
    return json.load(filename)


def string_to_color(in_col):
    new_col = in_col.replace('(', '')
    new_col = new_col.replace(')', '')
    rgb = new_col.split(', ')
    return int(rgb[0]), int(rgb[1]), int(rgb[2])