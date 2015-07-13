import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

j = 0

def waterfall(code, channels, tone, window, rate):

    def update(frame_number):
        global j
        data = np.asarray(code[0], np.complex)*mixer_sin
        j = (j+1)%height
        out[:,j] = abs(np.fft.rfft(np.asarray(code[0], np.complex)))
        im.set_data(np.fft.fftshift(out, axes=0).T)
        return im

    def init():
        out = np.empty((width, height), np.float)
        for i in range(height):
            out[:,i] = abs(np.fft.rfft(np.asarray(code[0], np.complex)))
        im.set_data(out)
        return im

    fig = plt.figure()

    plt.xlabel('Frequency Index')
    plt.ylabel('Time Index')

    data = np.asarray(code[0], np.complex)
    width = 1+len(data)/2
    height = width*3/4
    out = np.empty((width, height), np.float)
    for i in range(height):
        out[:,i] = abs(np.fft.rfft(np.asarray(code[0], np.complex)))

    if channels == 2:
        mixer_sin = np.array([(np.exp(2*np.pi*1j*tone*i/rate)) for i in range(len(data))])
    else:
        mixer_sin = np.array([(np.sin(2*np.pi*1 *tone*i/rate)) for i in range(len(data))])

    im = plt.imshow(
        np.fft.fftshift(out, axes=0).T,
        extent=(0-width/2, 0+width/2, 0, height),
        aspect='auto', interpolation='none', origin='lower')

    anim = animation.FuncAnimation(fig, update, interval=50,)
    #anim = animation.FuncAnimation(fig, update, init_func=init, interval=50, blit=True,)

    plt.show()

    return 0
