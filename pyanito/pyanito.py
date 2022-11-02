#from __future__ import print_function, division
import numpy as np # manipulate data
import sounddevice as sd
from scipy import signal # window signal

fs = 44100  # sampling rate Hz

### building blocks

def time_arr(tf, fs):
    """creates a time array with duration tf and sampling rate fs"""
    return np.linspace(0, tf, fs * tf)


def _round2int(x):
    return int(np.round(x))


round2int = np.vectorize(_round2int)


### Synthesisers

def sinewave(freq, time):
    """Creates sine wave"""
    y = np.sin(2 * np.pi * freq * time)
    #w = signal.tukey(len(y))  # window the signal
    return y #* w


def i2dec(n):
    """quadratic decay"""
    return np.arange(1, n+1)**2



def harmonic_i2dec(freq, time, n=5):
    """sine wave with n harmonics with quadratic decay
    time: ndarray"""
    quad_dec = np.arange(1, n+1)**2

    return harmonic(freq, time, n=n, decay=quad_dec)


def harmonic(freq, time, n=5, decay=None):
    """sine wave with n harmonics
    freq: num
        fundamental frequency
    time: ndarray
        time sampled
    n: number of harmonics
    decay: ndarray of size n
        Decay of the harmonics
        default None means no decay"""

    freqs = freq * np.arange(1, n+1)

    return experimental_harmony(freqs, time=time, decay=decay)


### play

def play(f, fs):
    """TODO: filter high frequencies or raise error when f > 800 Hz"""
    # if spectrum(f)
    return sd.play(f, fs)

def playwave(f, tf=1, timber=harmonic):
    """plays sine wave with frequency f and duration tf"""
    t = time_arr(tf, fs)
    y = timber(f, t)
    sd.play(y, fs)

#### Other handy sounds to play

def notesInSeries(freqs, time, synth=None):
    """play notes with frequencies freqs, each with a duration specified by time
    freqs: ndarray
        frequencies of the sine waves
    time: ndarray
        time
    synth: callable
        wave sythesizer, takes frequency and time
        default sinewave
    """
    if synth is None: # no decay of the harmonics
        synth = sinewave

    n = len(freqs)
    m = len(time)
    y = np.zeros(n*m)
    print(n, len(y))
    for i, freq in enumerate(freqs):
        yi = synth(freq, time)
        y[i*m:(i+1)*m] = yi

    return y


def experimental_harmony(freqs, time, decay=None, phase='random'):
    """sine waves with specific in frequencies (freqs)
    freqs: ndarray
        frequencies of the sine waves
    time: ndarray
    phase: ndarray
        None means no phase shift, phase = 0,
        default 'random' generates random phases"""
    n = len(freqs)

    if phase is None:
        phase = np.zeros(n)
    if phase == 'random':
        phase = np.random.rand(n)

    if decay is None: # no decay of the harmonics
        decay = np.ones(n)

    y = np.zeros_like(time)
    for i, freq in enumerate(freqs):
        y += np.sin(2 * np.pi * freq * time + phase[i]) / decay[i]

    w = signal.tukey(len(y))  # window the signal
    # print(np.isnan(y).any())

    return y * w


#### frequency, pitch and music theory

def key2frequency(n_key):
    """Returns the frequency given the key number"""
    return 440. * 2. ** ((n_key - 49.) / 12.)


keys2frequencies = np.vectorize(key2frequency)


def frequency2key(n_key):
    """Returns the frequency of the n-th key"""
    return 12. * np.log2(f / 440.) + 49.


frequencies2keys = np.vectorize(frequency2key)


def my_piano_key2frequency_fun(k1, f1, k2, f2):
    """Returns the linear function of callibrating the
    piano witth keys k1 and k2 with the respective frequencies
    f1 and f2
    Parameters
    ----------
    k1, f1: float
        key, frequency
        kf1: tuple
    k2, f2: float
        key, frequency
    Returns
    -------
    f: float
        frequency of n_key in the linear piano
    """
    b = (f1/f2)**(1/(k1-k2))
    a = f1/(b**(k1))
    f = lambda x :  a * b**(x)
    return f


def linear_piano_key2frequency(n_key):
    """Simulates the frequencies of a piano with linear intervals between notes
    Calibrate linear piano with A440, the 49th key and A880 the 61th key
    Parameters
    ----------
    n_key: int
        piano key
    Returns
    -------
    f: float
        frequency of n_key in the linear piano
    """
    f = 440. / 12. * n_key + 440. * (1. - 49. / 12.)
    return f



def linear_piano_key2frequency_fun(k1, f1, k2, f2):
    """Returns the linear function of callibrating the
    piano witth keys k1 and k2 with the respective frequencies
    f1 and f2
    Parameters
    ----------
    k1, f1: float
        key, frequency
        kf1: tuple
    k2, f2: float
        key, frequency
    Returns
    -------
    f: float
        frequency of n_key in the linear piano
    """
    m = (f2 - f1)/(k2 -k1)
    b = f1 - m * k1
    f = lambda x :  x * m + b
    return f


linear_piano_keys2frequencies = np.vectorize(linear_piano_key2frequency)


## Music theory


major_scale_intervals = np.array([2, 2, 1, 2, 2, 2, 1])


def major_scale_of_k(k, n):
    """Returns the n keys of the major scale starting at K"""
    keys = np.zeros(n+1)
    keys[0] = k
    for i in np.arange(n):
        keys[i+1] = keys[i] + major_scale_intervals[i%7] - 1

    return keys

def majorScaleKeys(n0=49, n_octaves=1):
    """Returns the keys of the major scale starting at n0
    TODO: generalise to more than one octave"""
    #intervals = np.array([2, 2, 1, 2, 2, 2, 1])
    intervals_from_key = np.cumsum(major_scale_intervals)
    keys = np.hstack((np.array([n0]), n0 + intervals_from_key))
    return keys


def majorScaleFreqs(n0=49, n_octaves=1):
    """Returns the frequencies of the major starting at the n0-th key"""
    keys = majorScaleKeys(n0=n0, n_octaves=n_octaves)
    return keys2frequencies(keys)


def linear_majorScaleFreqs(n0=49, n_octaves=1):
    """Returns the frequencies of the major starting at the n0-th key"""
    keys = majorScaleKeys(n0=n0, n_octaves=n_octaves)
    return linear_piano_key2frequency(keys)


def majorScaleFreqsFromFreq(f0=110, n=8):
    """Returns the frequencies of the major starting at the n0-th key"""
    keys = majorScaleKeys(n0=n0, n_octaves=n_octaves)
    return keys2frequencies(keys)

def pioanokey2note():
    """TODO returns notes A A# Bb B..."""
    return 0


def note2pianokey(note, octave=None, key=None):
    """return key index
    Parameters
    ----------
    note: str
        musical note Ab3, A3, A#, B, etc
    octave: int,
        default None, infer from the note, A#3 (last character), octave = 3
        if an integer is given, then is assumed that the notes are not passed with the octave.
        TODO: check possible errors here, eg. octave = 3 and note = A#3

    Example: note2pianokey('A#4') --> 50
    """
    assert isinstance(note, str), "note must be str"
    assert len(note) <= 3, "note should at most three characters long"
    if octave is None:
        octave = int(note[-1])
        n = note[:-1]
    else:
        n = note

    assert n[0] in "ABCDEFG", "note should be in A-G"
    assert octave in np.arange(8), "note octave should be int in 1-8"
    # mapping between notes and key numbers
    key2n = dict(zip(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], np.arange(1, 13)))
    key2n.update({'Bb': 11, 'Db': 2, 'Eb': 4, 'Gb': 7, 'Ab': 9})

    return 3 + 12 * (octave - 1) + key2n[n]


def note2F0(note):
    """Return frequency of the note"""
    k = note2pianokey(note)
    f = key2frequency(k)
    return f


def calibrate_piano(note_key1, note_key2):
    """TODO
    Simulates the frequencies of a piano with linear intervals between notes
    Calibrate linear piano with A440, A880
    TODO note_ket = (key, freq)"""
    assert isinstance(note_key1[0], int)
    assert isinstance(note_key2[0], int)
    assert isinstance(note_key1[1], float)
    assert isinstance(note_key2[1], float)
    return 0
    f1, k1 = note_key1
    f2, k2 = note_key2

    f = (f2 - f1) / (k2 - k1) * n_key + b

    return f