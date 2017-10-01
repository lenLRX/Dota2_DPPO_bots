from tkinter import *
from .Config import Config
from .simulator import DotaSimulator

master = Tk()
canvas = Canvas(master,width = Config.windows_size, height = Config.windows_size)
canvas.pack()
_engine = None
def _visualize():
    global canvas,_engine,master
    canvas = Canvas(master,width = Config.windows_size, height = Config.windows_size)
    eng = DotaSimulator(Config.dire_init_pos,canvas = canvas)
    canvas.pack()
    _engine = eng
    master.after(0,loop)
    master.mainloop()

def tk_main_loop(tup):
    global canvas,_engine
    fn, num_iter = tup
    fn.send(None)
    master.after(1, tk_main_loop, (fn, num_iter))

def visualize(fn, num_iter):
    global canvas,_engine,master
    canvas.delete("all")
    fn.send(None)
    fn.send((num_iter, canvas))
    master.after(1,tk_main_loop,(fn,num_iter))
    master.mainloop()

def loop():
    global canvas,_engine,master
    _engine.loop()
    _engine.draw()
    _engine.tick_tick()
    canvas.update_idletasks()
    master.after(1,loop)