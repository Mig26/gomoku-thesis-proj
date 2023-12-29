import sys
from tkinter import *
import gomoku
import filereader

game_instance = gomoku.GomokuGame(filereader.create_gomoku_game("consts.json"))

root = Tk()
root.geometry("170x232")
root.minsize(170, 232)
root.maxsize(170, 232)
root.title("Gomoku -- Main Menu")

style_numbers = ["georgia", 10, "white", 12, 2]

input_canvas = Canvas(root, relief="groove", borderwidth=0, highlightthickness=0)
input_canvas.grid(row=1, padx=2, pady=2)


def set_game_instance(new_instance):
    global game_instance
    game_instance = new_instance


def start_new_game():
    root.wm_state('iconic')
    gomoku.run(game_instance)
    game_over()


def game_over():
    root.wm_state('normal')


def quit_game():
    sys.exit()


button_1 = Button(input_canvas, text="New Game", bg=style_numbers[2], font=(style_numbers[0], style_numbers[1]), width=style_numbers[3], height=style_numbers[4], command=lambda: start_new_game())
button_1.grid(row=0, column=0, sticky="nsew")
button_2 = Button(input_canvas, text="Quit Game", bg=style_numbers[2], font=(style_numbers[0], style_numbers[1]), width=style_numbers[3], height=style_numbers[4], command=lambda: quit_game())
button_2.grid(row=1, column=0, sticky="nsew")


def mainmenu_run():
    root.mainloop()
