import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from termcolor import colored

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from teaming import logger


AGENTS=7
ROBOTS=4
ROWS=5
COLS=7

i=9

q=1
fname="tests/vary/"+str(AGENTS)+"-"+str(ROBOTS)+"-"+str(i)+"-"+str(q)+".pkl"


log = logger.logger()
#log.load("tests/evo38-5.pkl")
log.load(fname)

t=log.pull("test")
aprx=log.pull("aprx")
teams=log.pull("teams")[0]
index=log.pull("idxs")
print(len(teams))
def onclick(event):
    for i in range(10):
        print(" ")

    x=int(event.xdata)
    x=max(min(len(t)-1,x),0)
    arx=[y[1] for y in aprx[x]]
    arx=np.array(arx)
    for i in range(len(t[0])):

        val=np.round(t[x][i],3)
        if i in list(index[x]):
            print(colored(i,"yellow"),colored(val,"yellow"),end=" ")
        else:
            print(i,val,end=" ")
        print(teams[i],np.round(arx[i].T[0],2))


plt.plot(np.sum(t,axis=1))
btn='motion_notify_event'
#btn='button_press_event''
cid = plt.gcf().canvas.mpl_connect(btn, onclick)

plt.show()

