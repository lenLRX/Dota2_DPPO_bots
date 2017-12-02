from tkinter import *

windows_size = 600

master = Tk()
canvas = Canvas(master,width = windows_size, height = windows_size)
canvas.pack()

def tk_main_loop(tup):
    global canvas
    fn, num_iter = tup
    fn.send(None)
    master.after(1, tk_main_loop, (fn, num_iter))

def visualize(fn, num_iter):
    global canvas,master
    canvas.delete("all")
    fn.send((num_iter, canvas))
    master.after(1,tk_main_loop,(fn,num_iter))
    master.mainloop()