import sys
from tkinter import *
import gomoku
import filereader
import stats
from PIL import Image, ImageTk
from multiprocessing import Process


game_instance = gomoku.GomokuGame(filereader.create_gomoku_game("consts.json"))

root = Tk()
root.geometry("220x232")
root.minsize(220, 232)
root.maxsize(220, 232)
root.title("Gomoku -- Main Menu")
root.wm_iconphoto(False, ImageTk.PhotoImage(Image.open('res/ico.png')))

style_numbers = ["georgia", 10, "white", 12, 2]

input_canvas = Canvas(root, relief="groove", borderwidth=0, highlightthickness=0)
input_canvas.grid(row=1, padx=2, pady=2)
p1 = StringVar()
p2 = StringVar()
p1.set("Human")
p2.set("AI")
game_runs = StringVar()
game_runs.set("1")
delayvar = BooleanVar()
delayvar.set(True)


def set_player_type(playerid):
    if playerid == 0:
        newtype = p1.get()
    else:
        newtype = p2.get()
    gomoku.players[playerid].set_player(newtype, playerid)


def set_game_instance(new_instance):
    global game_instance
    game_instance = new_instance


def run():
    gomoku.run(game_instance)


def start_new_game():
    try:
        runs = int(game_runs.get())
        root.wm_state('iconic')
        stats.setup_logging(p1.get(), p2.get())
        game_instance.ai_delay = delayvar.get()
        for i in range(runs):
            stats.log_message(f"Game {i+1} begins.")
            game_instance.current_game = i+1
            game_instance.last_round = (i+1 == runs)
            gomoku.run(game_instance)
    except ValueError:
        print("Game runs value invalid.")
    game_over()


def game_over():
    root.wm_state('normal')
    game_instance.current_game = 0


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
gamerunslabel = Label(input_canvas, text="Number of games: ", font=(style_numbers[0], style_numbers[1]))
gamerunslabel.grid(row=5, column=0, sticky="w")
gamerunsentry = Entry(input_canvas, textvariable=game_runs)
gamerunsentry.grid(row=5, column=1, sticky="w")
delaybutton = Checkbutton(input_canvas, text="Use AI Delay", variable=delayvar, font=(style_numbers[0], style_numbers[1]))
delaybutton.grid(row=6, column=0, sticky="w")
button_2 = Button(input_canvas, text="Quit Game", bg=style_numbers[2], font=(style_numbers[0], style_numbers[1]), width=style_numbers[3], height=style_numbers[4], command=lambda: quit_game())
button_2.grid(row=7, column=0, sticky="nsew")


def mainmenu_run():
    root.mainloop()
