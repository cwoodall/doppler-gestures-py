#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotter(dump, channels, tone, window, rate):
    """
        Plots the fourier transform of dump
    """
    def update(frame_number, axis):
        res = mixer_sin * dump[0]
        rfft = abs(np.fft.rfft(res))
        axis.set_data(rfft_freqs,rfft)
        return axis

    length = dump.size
    rfft_freqs = np.fft.rfftfreq(length, d=1.0/rate)

    if channels == 2:
        mixer_sin = np.array([(np.exp(2*np.pi*1j*(tone-window/2)*i/rate)) for i in range(length)])
    else:
        mixer_sin = np.array([(np.sin(2*np.pi*(tone-window/2)*i/rate)) for i in range(length)])

    fig = plt.figure()
    ax = plt.axes(xlim=[0,window], ylim=[0,5*length])
    ax.grid(True)
    axis0 = ax.plot([],[])
    anim = animation.FuncAnimation(fig,update,
                                   fargs=(axis0),
                                   interval=50)

    plt.show()

    return 0
