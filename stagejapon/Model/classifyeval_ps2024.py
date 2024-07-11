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

datadir = DATAROOT + sounddata + '_ps_size' + str(IMAGESIZE) + '_samplefreq' + str(SAMPLEFREQ)
print('datadir: ', datadir)

# Setting name of directory for test
test_subdir = 'fold' + str(testdatano)
print('Test: '+test_subdir, end=' ', flush=True)

# Load test data
loaddata = np.load(datadir+'/'+test_subdir+'.npz')
test_data = loaddata['imagedata']
test_data = test_data.transpose(2,0,1)

# Mel scaling
if ftype=='m':
    for dataidx in range(test_data_amp.shape[0]):
        test_data[dataidx,:,:]=librosa.feature.melspectrogram(S=test_data[dataidx,:,:], sr=4096)

test_data = test_data / test_data.max() # Normalize
test_data[test_data<1e-8] = 1e-8 # Setting lower bound (0.00000001) to prevent log-of-zero
test_data = 10*np.log10(test_data) # Decibelize
test_data = test_data + 80 # Offset to make all values positive
#
test_data = test_data[:,:,:,np.newaxis]

test_label = loaddata['classid']
test_label_onehot = one_hot_encode(test_label)

# Load CNN model and fitted parameters
model = tf.keras.models.load_model(outputdir + 'trainedmodel_ps_size' + str(IMAGESIZE) + '_samplefreq' + str(SAMPLEFREQ) + '_' + fn_ftype + '.h5')

"""
pred_out = model.predict(test_data)
pred_label = np.argmax(pred_out,axis=1)

print(*pred_label, sep=',')
print(*test_label, sep=',')
"""
score, accuracy = model.evaluate(test_data, test_label_onehot, batch_size=32, verbose=1)
print("Accuracy: {:.1f}%".format(accuracy*100))

