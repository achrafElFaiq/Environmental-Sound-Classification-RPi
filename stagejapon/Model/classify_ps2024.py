# For intenship 2024
# What this program do: To predict class of environmental sounds (Urbansound8k / ESC50) using trained convolutional neural network (CNN).
# Input data: A collection of power spectrograms of the sounds.
# Output data: Predicted class

# libraries
import numpy as np
import librosa
import tensorflow as tf
import sklearn.metrics
import h5py
import math
import scipy as sp
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import os
import sys

# Conditions
IMAGESIZE = 128  # 128 / 256
SAMPLEFREQ = 8192  # 8192 / 16384 / 32768
DATAROOT = '../'
NNDROPOUT = 0.3
EPOCHS = 100

# Time Window
twin = sp.signal.windows.blackman(IMAGESIZE*4)

# Dataset
source = 'u' # 'u' for urbansound8k / 'e' for ESC50

# Frequency scale
ftype = 'l' # 'l' for Linear-scale / 'm' for Mel-scale

# Test folder (1 to 9)
testdatano = 1

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Checking condition parameters and setting output file names.
if source == 'u':
    sounddata = 'urbansound'
    DURATION = 4
    numclass = 10
    numfold = 10
elif source == 'e':
    sounddata = 'esc50'
    DURATION = 5
    numclass = 50
    numfold = 5
else:
    sys.exit('ERROR: Unknown sound data')

DATALENGTH = SAMPLEFREQ * DURATION

if ftype == 'l':
    fn_ftype = 'linearfreq'
    print('-> Linear frequency')
elif ftype == 'm':
    fn_ftype = 'melscale'
    print('-> Mel scale')
else:
    print('!! Invalid input type !!');
    sys.exit(1)

# Setting data directory name
outputdir = DATAROOT + "output_data_" + sounddata
if not(os.path.isdir(outputdir)):
    print('Output directory ' + outputdir + 'is not found.');
    sys.exit(1)
outputdir = outputdir + '/'

inputfile = sys.argv[1]
if not(os.path.isfile(inputfile)):
    print('Input sound file ' + inputfile + 'is not found.');
    sys.exit(1)

print('Sound file: ' + inputfile)

sig, s = librosa.load(inputfile, sr=SAMPLEFREQ, mono=True)
sig = np.resize(sig, SAMPLEFREQ*DURATION)

# Normalize
sig = sig / np.max(np.abs(sig))

step = math.floor((DATALENGTH-IMAGESIZE*4)/(IMAGESIZE-1))
spg = np.zeros(IMAGESIZE)
for timeframe in range(IMAGESIZE):
    subsig = sig[step*timeframe:step*timeframe+IMAGESIZE*4] * twin
    tmpps = np.square(np.abs(np.fft.fft(np.asarray(subsig))))
    spg = np.vstack([spg,tmpps[0:IMAGESIZE]])

spg = spg[1:,:]
spg = spg.T # Transpose

# Mel scaling
if ftype=='m':
    spg[:,:]=librosa.feature.melspectrogram(S=spg[:,:], sr=4096)

spg = spg / spg.max() # Normalize
spg[spg<1e-8] = 1e-8 # Setting lower bound (0.00000001) to prevent log-of-zero
spg = 10*np.log10(spg) # Decibelize
spg = spg + 80 # Offset to make all values positive
#
spg = spg[np.newaxis,:,:,np.newaxis]
# Load CNN model and fitted parameters
model = tf.keras.models.load_model(outputdir + 'trainedmodel_ps_size' + str(IMAGESIZE) + '_samplefreq' + str(SAMPLEFREQ) + '_' + fn_ftype + '.h5')

pred_out = model.predict(spg)
pred_label = np.argmax(pred_out,axis=1)

print('Predicted class: ' + str(pred_label))

