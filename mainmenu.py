import sys
from tkinter import *
from tkinter import ttk
import gomoku
import filereader
import stats
from PIL import Image, ImageTk
from multiprocessing import Process

WIDTH = 230
HEIGHT = 315
game_instance = gomoku.GomokuGame(filereader.create_gomoku_game("consts.json"))

root = Tk()
root.geometry(str(WIDTH) + "x" + str(HEIGHT))
root.minsize(WIDTH, HEIGHT)
root.maxsize(WIDTH, HEIGHT)
root.title("Gomoku -- Main Menu")
try:
    root.wm_iconphoto(False, ImageTk.PhotoImage(Image.open('res/ico.png')))
except TclError:
    pass

tabControl = ttk.Notebook(root)

tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)

tabControl.add(tab1, text='Play Game')
tabControl.add(tab2, text='Train')
tabControl.add(tab3, text='Replay')
tabControl.grid(row=0, sticky="w")

style_numbers = ["georgia", 10, "white", 12, 2]

input_canvas = Canvas(root, relief="groove", borderwidth=0, highlightthickness=0)
input_canvas.grid(row=1, padx=2, pady=2)
p1 = StringVar()
p2 = StringVar()
p1.set("Human")
p2.set("MM-AI")
game_runs = StringVar()
game_runs.set("1")
delayvar = BooleanVar()
delayvar.set(False)
logvar = BooleanVar()
logvar.set(False)
repvar = BooleanVar()
repvar.set(False)


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
        game_instance.ai_delay = delayvar.get()
        stats.should_log = logvar.get()
        stats.setup_logging(p1.get(), p2.get())
        for i in range(runs):
            stats.log_message(f"Game {i+1} begins.")
            game_instance.current_game = i+1
            game_instance.last_round = (i+1 == runs)
            gomoku.run(game_instance, i)
    except ValueError:
        print("Game runs value invalid.")
    game_over()


def game_over():
    root.wm_state('normal')
    game_instance.current_game = 0


def quit_game():
    sys.exit()


ttk.Label(tab1)

button_1 = Button(tab1, text="New Game", bg=style_numbers[2], font=(style_numbers[0], style_numbers[1]), width=style_numbers[3], height=style_numbers[4], command=lambda: start_new_game())
button_1.grid(row=0, column=0, sticky="w")
button_2 = Button(tab1, text="Train", bg=style_numbers[2], font=(style_numbers[0], style_numbers[1]), width=style_numbers[3], height=style_numbers[4], command=lambda: start_new_game())
button_2.grid(row=0, column=1, sticky="w")
player1typelabel = Label(tab1, text="Player 1", font=(style_numbers[0], style_numbers[1]))
player1typelabel.grid(row=2, column=0, sticky="w")
player2typelabel = Label(tab1, text="Player 2", font=(style_numbers[0], style_numbers[1]))
player2typelabel.grid(row=2, column=1, sticky="w")
radiobutton1 = Radiobutton(tab1, text="Human", variable=p1, value="Human", command=lambda: set_player_type(0))
radiobutton1.grid(row=3, column=0, sticky="w")
radiobutton2 = Radiobutton(tab1, text="AI", variable=p1, value="AI", command=lambda: set_player_type(0))
radiobutton2.grid(row=4, column=0, sticky="w")
radiobutton3 = Radiobutton(tab1, text="MM-AI", variable=p1, value="MM-AI", command=lambda: set_player_type(0))
radiobutton3.grid(row=5, column=0, sticky="w")
radiobutton4 = Radiobutton(tab1, text="Human", variable=p2, value="Human", command=lambda: set_player_type(1))
radiobutton4.grid(row=3, column=1, sticky="w")
radiobutton5 = Radiobutton(tab1, text="AI", variable=p2, value="AI", command=lambda: set_player_type(1))
radiobutton5.grid(row=4, column=1, sticky="w")
radiobutton6 = Radiobutton(tab1, text="MM-AI", variable=p2, value="MM-AI", command=lambda: set_player_type(1))
radiobutton6.grid(row=5, column=1, sticky="w")
gamerunslabel = Label(tab1, text="Number of games: ", font=(style_numbers[0], style_numbers[1]))
gamerunslabel.grid(row=6, column=0, sticky="w")
gamerunsentry = Entry(tab1, textvariable=game_runs)
gamerunsentry.grid(row=6, column=1, sticky="w")
delaybutton = Checkbutton(tab1, text="Use AI Delay", variable=delayvar, font=(style_numbers[0], style_numbers[1]))
delaybutton.grid(row=7, column=0, sticky="w")
logbutton = Checkbutton(tab1, text="Create log file", variable=logvar, font=(style_numbers[0], style_numbers[1]))
logbutton.grid(row=8, column=0, sticky="w")
replaybutton = Checkbutton(tab1, text="Save replays", variable=repvar, font=(style_numbers[0], style_numbers[1]))
replaybutton.grid(row=9, column=0, sticky="w")
button_3 = Button(input_canvas, text="Quit Game", bg=style_numbers[2], font=(style_numbers[0], style_numbers[1]), width=style_numbers[3], height=style_numbers[4], command=lambda: quit_game())
button_3.grid(row=1, column=0, sticky="e")

ttk.Label(tab2)
ttk.Label(tab3)

def mainmenu_run():
    root.mainloop()
