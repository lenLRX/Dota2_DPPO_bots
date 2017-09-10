from Tkinter import *
from Config import Config

master = Tk()

def visualize(Engine):
    canvas = Canvas(master,width = Config.windows_size, height = Config.windows_size)
    canvas.pack()
    while True:
        canvas.delete("all")
        Engine.draw(canvas)
        master.update_idletasks()
        master.update()