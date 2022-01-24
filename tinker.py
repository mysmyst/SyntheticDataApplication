from logging import root
from  tkinter import *
import tkinter
import PIL
from PIL import ImageTk
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

root =Tk()
root.title('Draw Tool')
root.geometry("500x500")

def graph():
    xmin = float(input("Enter lowest x value "))
    xmax = float(input("Enter highest x value "))
    ymin = float(input("Enter lowest y value "))
    ymax = float(input("Enter highest y value "))

    class LineBuilder:
        def __init__(self, line):
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            print('click', event)
            if event.inaxes!=self.line.axes: return
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

    fig, ax = plt.subplots()
    ax.set_title('click to build line segments')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    line, = ax.plot([], [])  # empty line
    linebuilder = LineBuilder(line)
    plt.show()
    x = line.get_xdata()
    y = line.get_ydata()
    print(x,y)  
    
bttn = Button(root, text="Test",command=graph)
bttn.pack()

root.mainloop()