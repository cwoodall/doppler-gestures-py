#!/usr/bin/env python

import ctypes
import logging
import sys
from collections import deque
from itertools import chain
from multiprocessing import Array, Event, Process, Queue
from struct import pack, unpack
import argparse

import numpy as np
import pyaudio

import pydoppler
from scipy import signal

TONE = 20000
WINDOW = 1000
CHANNELS = 2
CHUNK = 2048
RATE = 44100

FIR = True
HISTORY = True
PLOTTER = True
AMBIGUITY = False
WATERFALL = False

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
    stream = p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=RATE,
                    frames_per_buffer=RATE,
                    output=True,
                    input=False)

    stream.start_stream()
    sync.set()

    A = float(2**15 - 1)
    while 1:
        r = range(0, RATE)
        if CHANNELS == 2:
            L = [A*np.cos(2*np.pi*float(i)*float(freq)/RATE) for i in r]
        else:
            L = [A*np.sin(2*np.pi*float(i)*float(freq)/RATE) for i in r]
        R = [A*np.sin(2*np.pi*float(i)*float(freq)/RATE) for i in r]
        data = chain(*zip(L,R))
        chunk = b''.join(pack('<h', i) for i in data)
        stream.write(chunk)
    print("done")

    stream.stop_stream()
    stream.close()

    p.terminate()
    return True

def recorder(dump,freq, window_size, sync):
    """
    Records audio
    """
    p = pyaudio.PyAudio()

    freqs = np.fft.rfftfreq(CHUNK, d=1.0/RATE)
    display_freqs = (freq - window_size/2, freq + window_size/2)
    freq_range = np.where((freqs > display_freqs[0]) & (freqs<display_freqs[1]))
    frange = (freq_range[0][0],freq_range[0][-1])

    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    if FIR:
        fir = signal.firwin(64, [freq-window_size/2, freq+window_size/2], pass_zero=False, nyq=44100/2)
    stream.start_stream()

    frames = []

    sync.wait()

    i_history1 = deque(maxlen=3)
    i_history2 = deque(maxlen=3)
    coeffs = np.array([.1, .2, .7])

    while True:
        data = stream.read(CHUNK)
        frame = np.array(block2short(data), dtype=float)
        if CHANNELS == 2:
            frame.dtype = complex
        if FIR:
            frame = signal.convolve(frame, fir, 'same')
        np.copyto(dump, frame)

        if HISTORY:
            frame_fft = abs(np.fft.rfft(frame))
            freq_20khz_window = freqs[frange[0]:frange[1]]
            fft_20khz_window = frame_fft[frange[0]:frange[1]]
            fft_maxarg = np.argmax(fft_20khz_window)
            fft_peak = fft_20khz_window[fft_maxarg]

            # Lower Side Band
            for i, pwr in enumerate(reversed(fft_20khz_window[:fft_maxarg])):
                if pwr < fft_peak*.1:
                    i_history1.append(i)
                    if len(i_history1) >= i_history1.maxlen:
                        i_avg1 = sum(i_history1 * coeffs)
                        print "%10s" % "".join(["-" for h in range(10 if i_avg1 >= 10 else int(i_avg1))]),
                    break

            # Upper Side Band
            for i, pwr in enumerate(fft_20khz_window[fft_maxarg:]):
                if pwr < fft_peak*.1:
                    i_history2.append(i)
                    if len(i_history2) >= i_history2.maxlen:
                        i_avg2 = sum(i_history2 * coeffs)
                        print "%-10s" % "".join(["-" for h in range(10 if i_avg2 >= 10 else int(i_avg2))]),
                    break
            print

    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames

def main():
    """
    Doppler Gesture detector
    """
    global CHUNK
    global RATE
    global CHANNELS

    # Read in command-line arguments and switches
    parser = argparse.ArgumentParser(description='Plays a tone (20kHz default) and then looks for doppler shifts within a window range')
    parser.add_argument('--tone', '-t', dest='tone', action='store', type=int,
                        default=TONE, help='Tone (Hz)')
    parser.add_argument('--window', '-w', dest='window', action='store', type=int,
                        default=WINDOW, help='Window range (Hz)')
    parser.add_argument('--channels', '-c', dest='channels', action='store', type=int,
                        default=CHANNELS, help='Number of channels (1 or 2)')
    parser.add_argument('--size', '-s', dest='size', action='store', type=int,
                        default=CHUNK, help='Sample size')
    parser.add_argument('--rate', '-r', dest='rate', action='store', type=int,
                        default=RATE, help='Sample rate (Hz)')
    args = parser.parse_args()

    CHUNK = args.size
    RATE = args.rate
    CHANNELS = args.channels

    # Verify arguments

    # Check that the args.channels argument has the correct number of channels.
    if args.channels not in [1, 2]:
        print("Invalid number of channels. Please enter as 1 or 2")
        sys.exit(-1)

    if CHANNELS == 2:
        shared_array_base = Array(ctypes.c_double, 2*CHUNK)
    else:
        shared_array_base = Array(ctypes.c_double, CHUNK)

    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())

    if CHANNELS == 2:
        shared_array.dtype = complex

    shared_array = shared_array.reshape(1, CHUNK)

    sync_event = Event()

    # Initialize all processes and then start them
    tonePlayer_p = Process(target=tonePlayer, args=(
        args.tone,
        sync_event,))
    tonePlayer_p.daemon = True

    recorder_p = Process(target=recorder, args=(
        shared_array,
        args.tone,
        args.window,
        sync_event,))
    recorder_p.daemon = True

    if PLOTTER:
        plotter_p = Process(target=pydoppler.plotter, args=(
            shared_array,
            args.channels,
            args.tone,
            args.window,
            args.rate,))
        plotter_p.daemon = True

    if AMBIGUITY:
        ambiguity_p = Process(target=pydoppler.plotamb, args=(
            shared_array,
            args.channels,
            args.tone,
            args.window,
            args.rate,))
        ambiguity_p.daemon = True

    if WATERFALL:
        waterfall_p = Process(target=pydoppler.waterfall, args=(
            shared_array,
            args.channels,
            args.tone,
            args.window,
            args.rate,))
        waterfall_p.daemon = True

    recorder_p.start()
    tonePlayer_p.start()

    if PLOTTER:
        plotter_p.start()
    if AMBIGUITY:
        ambiguity_p.start()
    if WATERFALL:
        waterfall_p.start()

    tonePlayer_p.join()
    recorder_p.join()
    if PLOTTER:
        plotter_p.join()
    if AMBIGUITY:
        ambiguity_p.join()
    if WATERFALL:
        waterfall_p.join()

if __name__ == "__main__":
    main()
