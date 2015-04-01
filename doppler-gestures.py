#!/usr/bin/env python

from multiprocessing import Process, Queue, Event, Array
import ctypes
import pyaudio
from struct import pack, unpack
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
import sys
from collections import deque

def block2short(block):
    """
    Take a binary block produced by pyaudio and turn it into an array of
    shorts. Assumes the pyaudio.paInt16 datatype is being used.
    """
    # Each entry is 2 bytes long and block appears as a binary string (array 
    # of 1 byte characters). So the length of our final binary string is the
    # length of the block divided by 2.
    sample_len = len(block)/2
    fmt = "%dh" % (sample_len) # create the format string for unpacking
    return unpack(fmt, block)



def tonePlayer(freq, sync):
    """
    Plays a tone at frequency freq.
    """
    
    p = pyaudio.PyAudio()

    RATE  = 44100
    CHUNK = 1024*4
    A = (2**16 - 2)/2

    stream = p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=RATE,
                    frames_per_buffer=CHUNK,
                    output=True,
                    input=False)

    stream.start_stream()
    sync.set()
    h = 0
    s = 0
    while 1:
        L = [A*np.sin(2*np.pi*float(i)*float(freq)/RATE) for i in range(h*CHUNK, h*CHUNK + CHUNK)]
        R = [A*np.sin(2*np.pi*float(i)*float(freq)/RATE) for i in range(h*CHUNK, h*CHUNK + CHUNK)]
        data = chain(*zip(L,R))
        chunk = b''.join(pack('<h', i) for i in data)
        stream.write(chunk)
        h += 1
    print("done")

    stream.stop_stream()
    stream.close()

    p.terminate()
    return True

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

def recorder(dump,freq, window_size, sync):
    """
    Records audio
    """
    p = pyaudio.PyAudio()

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK= 1024*2
    freqs = np.fft.rfftfreq(CHUNK, d=1.0/RATE)
    display_freqs = (freq- window_size/2,freq + window_size/2)
    freq_range = np.where((freqs > display_freqs[0]) & (freqs<display_freqs[1]))
    frange = (freq_range[0][0],freq_range[0][-1])
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    fir = signal.firwin(64, [freq - window_size/2, freq + window_size/2], pass_zero=False,nyq=44100/2)
    stream.start_stream()
    
    frames = []

    sync.wait()
    i_history1 = deque(maxlen=3)
    i_history2 = deque(maxlen=3)
    coeffs = np.array([.1, .2, .7])
    while True:
        data = stream.read(CHUNK)
        frame = block2short(data)
        frame = signal.convolve(frame, fir, 'same')
        np.copyto(dump, frame)
        frame_fft = abs(np.fft.rfft(frame))
        freq_20khz_window = freqs[frange[0]:frange[1]]
        fft_20khz_window = frame_fft[frange[0]:frange[1]]
        fft_maxarg = np.argmax(fft_20khz_window) 
        fft_peak = fft_20khz_window[fft_maxarg]

        
        for i, pwr in enumerate(reversed(fft_20khz_window[:fft_maxarg])):
            if pwr < fft_peak*.1:
                i_history1.append(i)
                if len(i_history1) >= i_history1.maxlen:
                    i_avg1 = sum(i_history1 * coeffs)
                    print "%10s" % "".join(["-" for h in range(10 if i_avg1 >= 10 else int(i_avg1))]),
                break
        
        for i, pwr in enumerate(fft_20khz_window[fft_maxarg:]):
            if pwr < fft_peak*.1:
                i_history2.append(i)
                if len(i_history2) >= i_history2.maxlen:
                    i_avg2 = sum(i_history2 * coeffs)
                    print "%-10s" % "".join(["-" for h in range(10 if i_avg1 >= 10 else int(i_avg2))]),
                break
        print

#        print fft_peak
#       if fft_20khz_window[fft_maxarg] > 50000:
#          thresh= fft_20khz_window[fft_maxarg]*.12
#        else:
#            thresh = 55000

        #bw_freqs = freq_20khz_window[np.where(fft_20khz_window>thresh)[0]]
        #if bw_freqs.size > 2:
        #    bw = bw_freqs[-1] - bw_freqs[0]
        #    bw_lsb =  bw_freqs[0] - freq_20khz_window[fft_maxarg]
        #    bw_usb =  bw_freqs[-1] - freq_20khz_window[fft_maxarg]

          #  h = "".join([" " for i in range(int(80*(bw_usb + bw_lsb+ 200)/400))])
           # print (h+"|")

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return frames

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s freq freq_window" % sys.argv[0])
        sys.exit(-1)

    shared_array_base = Array(ctypes.c_double, 1024*2)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(1, 1024*2)    
    print shared_array
    
    s = Event()

    tonePlayer_p = Process(target=tonePlayer, args=(int(sys.argv[1]),s,))
    tonePlayer_p.daemon = True
    
    recorder_p = Process(target=recorder, args=(shared_array, int(sys.argv[1]),int(sys.argv[2]),s,))
    recorder_p.daemon = True

    plotter_p = Process(target=plotter, args=(shared_array,))
    plotter_p.daemon = True

    
    recorder_p.start()
    tonePlayer_p.start()
    plotter_p.start()

    tonePlayer_p.join()
    recorder_p.join()
    plotter_p.join()
