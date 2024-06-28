# For intenship 2024
# What this program do: To make input data (power spectrograms) for classification of environmental sounds (Urbansound8k / ESC50).
# Input data: Wave formatted environmental sound files.
# Output data: 10 or 5 collections of power spectrograms of the sounds.

# libraries
import numpy as np
import glob
import os
import os.path
import librosa
import math
import sys
import scipy as sp

# conditions
IMAGESIZE = 128

NOISERATE = 0
SAMPLEFREQ = 8192  # 8192 / 16384 / 32768

DATAROOT = '../'

# Time Window
twin = sp.signal.windows.blackman(IMAGESIZE*4)

# Dataset
source = 'u' # 'u' for urbansound8k / 'e'  for ESC50

if source == 'u':
    SOUNDDIR = DATAROOT+'urbansound/'
    sounddata = 'urbansound'
    DURATION = 4
    numclass = 10
    numfold = 10
    foldname = ['fold'+str(i)+'/' for i in range(1,numfold+1)]
elif source == 'e':
    SOUNDDIR = DATAROOT+'esc50data/audio/'
    sounddata = 'esc50'
    DURATION = 5
    numclass = 50
    numfold = 5
    foldname = [str(i)+'-' for i in range(1,5+1)]
else:
    sys.exit('ERROR: Unknown sound data')

DATALENGTH = SAMPLEFREQ * DURATION

save_dir = DATAROOT + sounddata + '_ps_size' + str(IMAGESIZE) + '_samplefreq' + str(SAMPLEFREQ) +'/'

if os.path.isdir(save_dir):
    print('save_dir "'+save_dir+'" already exists')
    if input('Over write OK? [y/n] > ') == 'y':
        print('"y" is selected.')
    else:
        sys.exit('EXIT')
else:
    os.mkdir(save_dir)

for foldno in range(numfold):
    datapath = SOUNDDIR+foldname[foldno]
    file_ext="*.wav"
    print(datapath.split('/')[-2]+datapath.split('/')[-1])
    spgs = np.zeros([IMAGESIZE,IMAGESIZE])
    labels = 0
    # Processing for every wav-file in the target directory
    k=0
    for fn in glob.glob(datapath + file_ext):
        sig, s = librosa.load(fn, sr=SAMPLEFREQ, mono=True)
        sig = np.resize(sig, SAMPLEFREQ*DURATION)

        # Class ID (Label)
        if source=='u':
            label = fn.split('/')[-1].split('-')[1]
        elif source=='e':
            label = fn.split('/')[-1].split('-')[-1].split('.')[0]
        else:
            sys.exit('ERROR: Unknown sound data')

        k = k + 1
        if k%100 == 0:
            print('o', end='', flush=True)
        elif k%10 == 0:
            print('.', end='', flush=True)

        # Normalize
        sig = sig / np.max(np.abs(sig))

        step = math.floor((DATALENGTH-IMAGESIZE*4)/(IMAGESIZE-1))
        spg = np.zeros(IMAGESIZE)
        for timeframe in range(IMAGESIZE):
            subsig = sig[step*timeframe:step*timeframe+IMAGESIZE*4] * twin
            #subsig = subsig.reshape(-1,1)
            tmpps = np.square(np.abs(np.fft.fft(np.asarray(subsig))))
            spg = np.vstack([spg,tmpps[0:IMAGESIZE]])

        spg = spg[1:,:]
        spg = spg.T # Transpose

        spgs=np.dstack([spgs, spg])
        labels=np.append(labels,label)

    spgs = spgs[:,:,1:]
    labels = labels[1:]
    print(' ')

    np.savez_compressed(save_dir+'fold'+str(foldno+1)+'.npz', imagedata=np.array(spgs), classid=np.array(labels,dtype = int))

    del spgs, labels
