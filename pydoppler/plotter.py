import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotter(dump):
    """
        Plots the fourier transform of dump
    """
    def update(frame_number, axis):
        res = mixer_sin * dump[0]
        rfft = abs(np.fft.rfft(res))
        axis.set_data(rfft_freqs,rfft)
        return axis

    f = 20000
    mixer_sin = np.array([(np.sin(2*np.pi*(f-1000)*i/44100)) for i in range(1024*2)])
    rfft_freqs = np.fft.rfftfreq(1024*2, d=1.0/44100)

    fig = plt.figure()
    ax = plt.axes(xlim=[0,2000], ylim=[0,1024*10])
    axis0 = ax.plot([],[])
    anim = animation.FuncAnimation(fig,update,
                                   fargs=(axis0),
                                   interval=50)

    plt.show()

    return 0
