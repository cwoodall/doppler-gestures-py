import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
FREQ = 1000
RATE = 44100
CHUNK = 1024


h= 3


def update(frameNum, axis):
    data = [(np.sin(2*np.pi*10*i/1000) + 1) * 1023/2 for i in range(frameNum, frameNum+10)]
    axis.set_data(range(len(data)), data)
    return axis

fig = plt.figure()
ax = plt.axes(xlim=(0,10), ylim=(0,1023))
axis0 = ax.plot([],[])
anim = animation.FuncAnimation(fig,update,
                               fargs=(axis0),
                               interval=50)

plt.show()

analogPlot.close()

print('exiting.')


