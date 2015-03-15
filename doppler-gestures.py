#!/usr/bin/env python

from multiprocessing import Process, Queue, Event
import pyaudio
import wave
import time
import sys
import struct
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
    return struct.unpack(fmt, block)



def tonePlayer(tone_file,sync):
    wf = wave.open(tone_file, 'rb')
    p = pyaudio.PyAudio()

    CHUNK=1024

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    frames_per_buffer=CHUNK,
                    output=True,
                    input=False)

    stream.start_stream()
    time.sleep(1)

    for i in range(1000):
        data = wf.readframes(CHUNK)
        sync.set()
        while data != '':
            stream.write(data)
            data = wf.readframes(CHUNK)
        wf.rewind()
    print("done")
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()
    return True

def recorder(q,sync):
    p = pyaudio.PyAudio()

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK= 1024*2
    freq = np.fft.rfftfreq(CHUNK, d=1.0/RATE)
    display_freqs = (19000,21000)
    freq_range = np.where((freq > display_freqs[0]) & (freq<display_freqs[1]))
    frange = (freq_range[0][0],freq_range[0][-1])
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    fir = signal.firwin(64, [19500, 20500], pass_zero=False,nyq=44100/2)
    stream.start_stream()
    
    frames = []
    #plt.ion()
    #plt.show()
    #plt.draw()

    sync.wait()
    
    for i in range(100000):
        data = stream.read(CHUNK)
        frame = block2short(data)
        frame = signal.convolve(frame, fir, 'same')
        frame_fft = abs(np.fft.rfft(frame))
        freq_20khz_window = freq[frange[0]:frange[1]]
        fft_20khz_window = frame_fft[frange[0]:frange[1]]
        fft_maxarg = np.argmax(fft_20khz_window) 
        
        if fft_20khz_window[fft_maxarg] > 50000:
            thresh= fft_20khz_window[fft_maxarg]*.12
        else:
            thresh = 55000
 #       plt.clf()
  #      plt.plot(freq[frange[0]:frange[1]], [thresh for i in range(len(fft_20khz_window))])
   #     plt.plot(freq[frange[0]:frange[1]], fft_20khz_window)
        bw_freqs = freq_20khz_window[np.where(fft_20khz_window>thresh)[0]]
        if bw_freqs.size > 2:
            bw = bw_freqs[-1] - bw_freqs[0]
            bw_lsb =  bw_freqs[0] - freq_20khz_window[fft_maxarg]
            bw_usb =  bw_freqs[-1] - freq_20khz_window[fft_maxarg]

            h = "".join([" " for i in range(int(80*(bw_usb + bw_lsb+ 200)/400))])
            print (h+"|")
    #    plt.ylim([0,CHUNK**2 / 2])
        
     #   plt.draw()


    print "DONE"
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return frames

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)
    for i in range(10):
        q = Queue()
        s = Event()
        tonePlayer_p = Process(target=tonePlayer, args=(sys.argv[1],s,))
        tonePlayer_p.daemon = True

        recorder_p = Process(target=recorder, args=(q,s,))
        recorder_p.daemon = True

        recorder_p.start()
        tonePlayer_p.start()


        tonePlayer_p.join()
        recorder_p.join()
        time.sleep(1)

#print rec


