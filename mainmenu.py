import sys
from tkinter import *
import gomoku
import filereader
from PIL import Image, ImageTk
from multiprocessing import Process


game_instance = gomoku.GomokuGame(filereader.create_gomoku_game("consts.json"))

root = Tk()
root.geometry("170x232")
root.minsize(170, 232)
root.maxsize(170, 232)
root.title("Gomoku -- Main Menu")
root.wm_iconphoto(False, ImageTk.PhotoImage(Image.open('res/ico.png')))

style_numbers = ["georgia", 10, "white", 12, 2]

input_canvas = Canvas(root, relief="groove", borderwidth=0, highlightthickness=0)
input_canvas.grid(row=1, padx=2, pady=2)
p1 = StringVar()
p2 = StringVar()
p1.set("Human")
p2.set("AI")


def set_player_type(playerid):
    if playerid == 0:
        newtype = p1.get()
    else:
        newtype = p2.get()
    gomoku.players[playerid].set_player(newtype, playerid)
    print(newtype)

def set_game_instance(new_instance):
    global game_instance
    game_instance = new_instance


def run():
    gomoku.run(game_instance)


def start_new_game():
    # gomoku.player1.set_player("Human", 0)
    # gomoku.player2.set_player("AI", 1)
    root.wm_state('iconic')
    gomoku.run(game_instance)
    game_over()


def game_over():
    root.wm_state('normal')


def quit_game():
    sys.exit()


button_1 = Button(input_canvas, text="New Game", bg=style_numbers[2], font=(style_numbers[0], style_numbers[1]), width=style_numbers[3], height=style_numbers[4], command=lambda: start_new_game())
button_1.grid(row=0, column=0, sticky="nsew")
player1typelabel = Label(input_canvas, text="Player 1", font=(style_numbers[0], style_numbers[1]))
player1typelabel.grid(row=2, column=0, sticky="w")
player2typelabel = Label(input_canvas, text="Player 2", font=(style_numbers[0], style_numbers[1]))
player2typelabel.grid(row=2, column=1, sticky="w")
radiobutton1 = Radiobutton(input_canvas, text="Human", variable=p1, value="Human", command=lambda: set_player_type(0))
radiobutton1.grid(row=3, column=0, sticky="w")
radiobutton2 = Radiobutton(input_canvas, text="AI", variable=p1, value="AI", command=lambda: set_player_type(0))
radiobutton2.grid(row=4, column=0, sticky="w")
radiobutton3 = Radiobutton(input_canvas, text="Human", variable=p2, value="Human", command=lambda: set_player_type(1))
radiobutton3.grid(row=3, column=1, sticky="w")
radiobutton4 = Radiobutton(input_canvas, text="AI", variable=p2, value="AI", command=lambda: set_player_type(1))
radiobutton4.grid(row=4, column=1, sticky="w")
button_2 = Button(input_canvas, text="Quit Game", bg=style_numbers[2], font=(style_numbers[0], style_numbers[1]), width=style_numbers[3], height=style_numbers[4], command=lambda: quit_game())
button_2.grid(row=5, column=0, sticky="nsew")


def mainmenu_run():
    root.mainloop()
