from logging import root
from  tkinter import *
import tkinter
import PIL
from PIL import ImageTk
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt

root =Tk()
root.title('Draw Tool')
root.geometry("500x500")
xmin = float(input("Enter lowest x value "))
xmax = float(input("Enter highest x value "))
ymin = float(input("Enter lowest y value "))
ymax = float(input("Enter highest y value "))

def graph():
    
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
    X = line.get_xdata()
    y = line.get_ydata()
    return(X,y)


# X, y = graph()

def equation(X, y):
    X_diff, y_diff, m, c, Range, R_total = list(), list(), list(), list(), dict(), 0
    for i in range(1,len(X)):
        X_diff.append(float(X[i]-X[i-1]))
        Range[i] = float(X_diff[i-1])
        R_total += float(X_diff[i-1])
        y_diff.append(float(y[i]-y[i-1]))
        slope = float(y_diff[i-1]/X_diff[i-1])
        m.append(float(slope))
        c.append(float(y[i-1]-m[i-1]*X[i-1]))
    Range = sorted(Range.items(), key = lambda item: item[1], reverse=True )
    return(X, y, X_diff,y_diff, m, c, Range, R_total)

# X, y, X_diff,y_diff, m, c, Range, R_total = equation(X,y)
# n = int(input("Number of entries: "))

def point_division(n, Range, R_total):
    number, num_sum = dict(), 0
    for key, item in Range:
        temp = math.floor(item/R_total*n)
        number[key] = temp
        num_sum += temp
    n_median = n-num_sum
    return(number, n_median)

# number, n_median = point_division(n, Range, R_total)
def change_range(a,b,c,d,t):
    ft = list()
    for i in range(0, len(t)):
        (b-a)==0
        ft.append(float(c + float((float((d-c))*float((t[i]-a)))/(float(b-a)))))
    return ft

# distribution = input('Enter distribution name: ')
def points(entries, lower_bound, upper_bound, distribution):
    generated = np.ones(entries)
    distribution = distribution.strip()
    if distribution.lower() == 'random':
        generated = np.random.rand(entries)
        generated = change_range(np.min(generated), np.max(generated), lower_bound, upper_bound, generated)
    elif distribution.lower() == 'normal':
        generated = np.random.randn(entries)
        generated = change_range(np.min(generated), np.max(generated), lower_bound, upper_bound, generated)
    elif distribution.lower() == 'random-integer':
        generated = np.random.randint(low=math.floor(lower_bound),high=math.floor(upper_bound), size=entries)
    elif distribution.lower() == 'beta':
        alpha = float(input("Enter alpha value for beta distribution function: "))
        if alpha<=0:
            alpha=0.5
        beta = float(input("Enter beta value for beta distribution function: "))
        if beta<=0:
            beta=0.5
        generated = np.random.beta(a=alpha, b=beta, size=entries)
        generated = change_range(np.min(generated), np.max(generated), lower_bound, upper_bound, generated)
    elif distribution.lower() == 'binomial':
        trials = int(input("Enter the number of trials performed: "))
        prob = float(input("Enter the probability of success: "))
        if prob<0 or prob>1:
            prob=0.5
        generated = np.random.binomial(n=trials, p=prob, size=entries)
        generated = change_range(np.min(generated), np.max(generated), lower_bound, upper_bound, generated)
    elif distribution.lower() == 'chi-square':
        df = int(input("Enter the degrees of freedom: "))
        if df < 0:
            df=0.5
        generated = np.random.chisquare(df=df, size=entries)
        generated = change_range(np.min(generated), np.max(generated), lower_bound, upper_bound, generated)
    return generated

def generate_column(number, n_median, X, distribution, m,c):
    Xlist = list()
    generated_column = list()
    for key, item in number.items():
        input = points(item, X[key-1], X[key], distribution)
        Xlist = np.concatenate([Xlist, input], axis=0)
        for i in range(0, len(input)):
            generated_column.append(float(m[key-1] * float(input[i]) + c[key-1]))
    if n_median >0:
        med_val = np.median(Xlist)
        op = 0 
        med = np.array([med_val]*n_median)
        for i in range(1,len(X)):
            if X[i] >= med_val and X[i-1] <= med_val:
                op = m[i-1] * med_val + c[i-1] 
        Xlist = np.concatenate([Xlist, med], axis=0)
        generated_column+=[float(op)]*n_median
    return (list(Xlist),generated_column)

# input, generated_column = generate_column(number, n_median, X, distribution, m, c)
# print(input, generated_column) 

# def data_relationship(generated_columns,relationship_type, ):

def main_generation():
    X, y = graph()
    X, y, X_diff,y_diff, m, c, Range, R_total = equation(X,y)
    n = int(input("Number of entries: "))
    number, n_median = point_division(n, Range, R_total)
    distribution = input('Enter distribution name: ')
    genX, generated_column = generate_column(number, n_median, X, distribution, m, c)
    print(genX, generated_column) 

main_generation()

bttn = Button(root, text="Test",command=graph)
bttn.pack()

root.mainloop()