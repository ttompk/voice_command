### Audio transformation

# reference:
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/simple_audio.ipynb#scrollTo=0SQl8yXl3kNP
'''
Training data specs:
    16-bit system, range from -32768 to 32767
    sample rate = 16kHz 

Steps:
1. read in data
2. convert binary audio file to waveform 
3. 
'''

import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

def decode_audio(audio_binary):
    '''
    'tf.audio.decode_wav' returns WAV-encoded audio as a Tensor and the sample rate
    'tf.audio.decode_wav' normalizes values to range [-1.0, 1.0]
    'tf.squeeze' 
    '''
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)  #tf.squeeze() reshapes the data  

# read in binary file
audio_binary = tf.io.read_file(file_path)

# return audio file as tensor
waveform = decode_audio(audio_binary)

def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(waveform, tf.float32)  # changes type of tensor to float32
    equal_length = tf.concat([waveform, zero_padding], 0)
    
    # 
    ''' tf.signal.stft()
    Splits the signal into windows of time and runs a 
    Fourier transform on each window, preserving some time 
    information, and returning a 2D tensor that you can run 
    standard convolutions on.'''
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

AUTOTUNE = tf.data.AUTOTUNE
spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
