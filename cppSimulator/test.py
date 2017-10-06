from tkinter import *
from cppSimulator import *

master = Tk()
canvas = Canvas(master,width = 400, height = 400)
canvas.pack()

sim = cppSimulator(canvas)

def tk_main_loop():
    print(sim.get_time())
    sim.loop()
    master.after(1, tk_main_loop)
    print(sim.get_state_tup("Radiant",0))

master.after(1,tk_main_loop)
master.mainloop()