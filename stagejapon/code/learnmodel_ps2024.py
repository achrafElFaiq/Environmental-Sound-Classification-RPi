# For intenship 2024
# What this program do: To train a convolutional neural network (CNN) for classification of environmental sounds (Urbansound8k / ESC50).
# Input data: 10 collections of power spectrograms of the sounds.
# Output data: Traind CNN.

# libraries
import numpy as np
import librosa
import tensorflow as tf
import sklearn.metrics
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import os
import sys

DATAROOT = '../'

# Conditions
IMAGESIZE = 128  # 128 / 256
SAMPLEFREQ = 8192  # 8192 / 16384 / 32768
NNDROPOUT = 0.3
EPOCHS = 100

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
    print('ERROR: Unknown sound data')
    sis.exit(1)

if ftype == 'l':
    fn_ftype = 'linearfreq'
    print('-> Linear frequency')
elif ftype == 'm':
    fn_ftype = 'melscale'
    print('-> Mel scale')
else:
    print('ERROR: Unknown frequency scale')
    sys.exit(1)

# Setting data directory name
datadir = DATAROOT + sounddata + '_ps_size' + str(IMAGESIZE) + '_samplefreq' + str(SAMPLEFREQ)
print('datadir: ', datadir)

outputdir = DATAROOT + "output_data_" + sounddata
if not(os.path.isdir(outputdir)):
    os.mkdir(outputdir)
outputdir = outputdir + '/'

# Making an array of data directory name.
subdirs=['fold1']
for foldno in range(2,numfold+1):
    subdirs.append('fold'+str(foldno))

# Setting name of directory for test
test_subdir = 'fold' + str(testdatano)
print('Test: '+test_subdir, end=' ', flush=True)

# Setting name of directory for training
train_subdirs = subdirs
#train_subdirs.remove(test_subdir)

# Load training data
train_data = np.zeros([IMAGESIZE,IMAGESIZE])
train_label = 0
for subdir in train_subdirs:
    loaddata = np.load(datadir+'/'+subdir+'.npz')
    train_data = np.dstack([train_data, loaddata['imagedata']])
    train_label = np.append(train_label,loaddata['classid'])

train_data = train_data[:,:,1:]
train_data = train_data.transpose(2,0,1)

# Mel scaling
if ftype=='m':
    for dataidx in range(train_data_amp.shape[0]):
        train_data[dataidx,:,:]=librosa.feature.melspectrogram(S=train_data[dataidx,:,:], sr=4096)

train_data = train_data / train_data.max() # Normalize
train_data[train_data<1e-8] = 1e-8 # Setting lower bound (0.00000001) to prevent log-of-zero
train_data = 10*np.log10(train_data) # Decibelize
train_data = train_data + 80 # Offset to make all values positive
#
train_data = train_data[:,:,:,np.newaxis]
#
train_label = train_label[1:]
train_label_onehot = one_hot_encode(train_label)

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
#
test_label = loaddata['classid']
test_label_onehot = one_hot_encode(test_label)

# Convolutional neural network
model = tf.keras.models.Sequential()

ishape=(IMAGESIZE, IMAGESIZE, 1)

# Conv 1
model.add(tf.keras.layers.Conv2D(12, (3,3), padding='same', input_shape=ishape))
# BatchNormalization
model.add(tf.keras.layers.BatchNormalization())
# Activation:relu
model.add(tf.keras.layers.Activation('relu'))
# Max pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))

# Conv 2
model.add(tf.keras.layers.Conv2D(24, (3,3), padding='same'))
# BatchNormalization
model.add(tf.keras.layers.BatchNormalization())
# Activation:relu
model.add(tf.keras.layers.Activation('relu'))
# Max pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))

# Conv 3
model.add(tf.keras.layers.Conv2D(48, (3,3), padding='same'))
# BatchNormalization
model.add(tf.keras.layers.BatchNormalization())
# Activation:relu
model.add(tf.keras.layers.Activation('relu'))
# Max pooling
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Conv 4
model.add(tf.keras.layers.Conv2D(48, (3,3), padding='same'))
# BatchNormalization
model.add(tf.keras.layers.BatchNormalization())
# Activation:relu
model.add(tf.keras.layers.Activation('relu'))

# Flatten
model.add(tf.keras.layers.Flatten())

# Fully connected 1
model.add(tf.keras.layers.Dense(64, kernel_regularizer=l2(0.001)))
# Activation:relu
model.add(tf.keras.layers.Activation('relu'))
# Dropout
model.add(tf.keras.layers.Dropout(NNDROPOUT))

# Output
model.add(tf.keras.layers.Dense(numclass, kernel_regularizer=l2(0.001)))
# Dropout
model.add(tf.keras.layers.Dropout(NNDROPOUT))
# Activation:softmax
model.add(tf.keras.layers.Activation('softmax'))

# Adam(Amsgrad)
optim = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)

history = model.fit(train_data, train_label_onehot, batch_size=16, epochs=EPOCHS, verbose=1)

pred_out = model.predict(test_data)
pred_label = np.argmax(pred_out,axis=1)

cm = sklearn.metrics.confusion_matrix(pred_label, test_label, labels=list(range(numclass)))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

np.set_printoptions(formatter={'float': '{:3.0f}'.format})
#print(cm_normalized*100)

score, accuracy = model.evaluate(test_data, test_label_onehot, batch_size=32, verbose=1)
print("Accuracy: {:.1f}%".format(accuracy*100))

# Save CNN model and fitted parameters
model.save(outputdir + 'trainedmodel_ps_size' + str(IMAGESIZE) + '_samplefreq' + str(SAMPLEFREQ) + '_' + fn_ftype + '.h5')