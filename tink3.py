from logging import root
from  tkinter import *
import tkinter
import PIL
import pandas as pd
from PIL import ImageTk
from PIL import Image
import math
from scipy.stats import pearsonr,spearmanr, tstd
from pandas.core.arrays.sparse import dtype
import numpy as np
import matplotlib.pyplot as plt
import sys

root =Tk()
root.title('Draw Tool')
root.geometry("500x500")

def graph():
    xmin = float(sys.argv[0])
    xmax = float(sys.argv[1])
    ymin = float(sys.argv[2])
    ymax = float(sys.argv[3])
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

bttn = Button(root, text="Test",command=graph)
bttn.pack()

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
        if (b-a) ==0:
            ft.append(float(c + float((float((d-c))*float((t[i]-a)))/(float(1)))))
        else:
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

def main_generation():
    X, y = graph()
    X, y, X_diff,y_diff, m, c, Range, R_total = equation(X,y)
    n = int(input("Number of entries: "))
    number, n_median = point_division(n, Range, R_total)
    distribution = input('Enter distribution name: ')
    genX, generated_column = generate_column(number, n_median, X, distribution, m, c)
    return(X, y, genX, generated_column)

def calc_relation(relation):
    if relation==1:
        relX, relY = graph()
        corr = pearsonr(relX, relY)[0]
        std_dev_relX = tstd(relX)
        std_dev_relY = tstd(relY)
        covxy = corr*std_dev_relX*std_dev_relY
    elif relation==2:
        relX, relY = graph()
        corr = spearmanr(relX, relY)[0]
        std_dev_relX = tstd(relX)
        std_dev_relY = tstd(relY)
        covxy = corr*std_dev_relX*std_dev_relY
    return covxy

def relationship(generated_columns,relationship_type):
    covxy = list()
    variances = generated_columns.var(axis=0)
    mean = generated_columns.mean(axis=0)
    for i in generated_columns.columns:
        generated_columns[i] = generated_columns[i]-mean[i]
    for j in range(0,len(relationship_type)):
        covxy.append(calc_relation(relationship_type[j]))
    Cov_var_matrix = generated_columns.cov()
    Cov_var_matrix = Cov_var_matrix.to_numpy()
    count = 0
    for i in generated_columns.columns:
        Cov_var_matrix[count][count] = variances[i]
        count+=1
    covxy = np.array(covxy)
    l = len(generated_columns.columns)
    count=0
    # covxy = covxy.reshape((l-1,l-1))
    for i in range(0, l):
        for j in range(0, l):
            if j<i:
                Cov_var_matrix[i][j]=covxy[count]
                Cov_var_matrix[j][i]=covxy[count]
                count+=1
    values, vectors = np.linalg.eig(Cov_var_matrix)
    P = np.dot(vectors.T, generated_columns.T)
    related_generated_columns = P.T
    related_generated_columns = pd.DataFrame(data = related_generated_columns, columns=['Col1','Col2'])
    for i in related_generated_columns.columns:
        related_generated_columns[i] = related_generated_columns[i]+mean[i]
    return(related_generated_columns)

def main_func():
    X1, y1, genX1, genY1 = main_generation()
    X2, y2, genX2, genY2 = main_generation()
    col = np.array(list(zip(genY1,genY2)))
    generated_columns = pd.DataFrame(data=col, columns=['Col1','Col2'])
    print(relationship(generated_columns,[1]))

main_func()