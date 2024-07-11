# Project: Training CNN Model on Ubuntu with GPU and application using Raspberry GUI App

## Introduction
This project is developed as part of our internship research topic at OYAMA INSTITUTE OF TECHNOLOGY under the guidance of Dr. HIRATA Katsumi. It focuses on creating a GUI application for the Raspberry Pi 4 that uses a Convolutional Neural Network (CNN) model to classify environmental sounds based on their spectrograms. The project involves setting up the necessary software environment on Ubuntu 20.04.6 LTS, configuring GPU support, and developing the model and the GUI application.

## Table of Contents
1. [Machine](#machine)
   - [Installing Ubuntu (20.04.6 LTS)](#installing-latest-version-of-ubuntu-20046-lts)
   - [Configuring Wired Connection on Ubuntu (If You Don't Have WIFI)](#configuring-wired-connection-on-ubuntu-if-you-dont-have-wifi)
   - [Setup Environment for GPU Usage](#setup-environment-for-gpu-usage)
     - [Local Setup](#local-setup)
     - [Virtual Environment Setup (Python 3.9.19)](#virtual-environment-setup-python-3810)
   - [Model](#model)
     - [Data](#data)
     - [Data Preprocessing](#data-preprocessing)
     - [Model Conception](#model-conception)
     - [Model Training](#model-training)
     - [Model Analysis](#model-analysis)
2. [Raspberry](#raspberry)
   - [Using the GUI app](#using-the-gui-app)


## Machine

### Installing <u>Ubuntu 22.04.4 LTS (Jammy Jellyfish)</u>
To set up your machine for this project, you need to install the latest version of Ubuntu, which is 22.04.4 LTS (Jammy Jellyfish). Follow the steps below:

1. **Download Ubuntu 22.04.4 LTS (Jammy Jellyfish)**
   - From the official website: [Ubuntu 22.04.4 LTS](https://releases.ubuntu.com/jammy/)

2. **Create a Bootable USB**
   - Use Rufus for Windows or Balena Etcher for Mac to make your USB bootable with the downloaded ISO file.

3. **Install Ubuntu**
   - Plug the USB into the target machine and start it in BIOS mode (typically by pressing the "Del" key during startup, but this can vary).
   - Set the primary boot device to the USB.
   - Follow the on-screen instructions to install the OS on your machine.

#### Additional Steps
   - After the installation is complete, remove the USB drive and restart the machine.
   - Follow any additional on-screen prompts to complete the setup, including creating a user account and setting up your network connection.

By following these steps, you will have the focal version of Ubuntu installed and ready for the next stages of your project setup.


### Configuring Wired Connection on Ubuntu (If You Don't Have WIFI)

1. **Identify Network Interface**
   - Use the command `ip link show` or `ifconfig` to find the name of your network interface.

2. **Create WPA Supplicant Configuration**
   - Create a `wpa_supplicant.conf` file in `/etc/wpa_supplicant/`. (A sample file will be in the repository)

3. **Run WPA Supplicant**
   - Use the command:
     ```sh
     sudo wpa_supplicant -Dwired -ieth0 -c /etc/wpa_supplicant/wpa_supplicant.conf
     ```
     (Replace `eth0` with your network interface name).

### Setup Environment for GPU Usage

#### Local Setup

1. **Install GPU Drivers**
   - GPU drivers might be provided with the OS.

2. **Install CUDA, CUDA Toolkits, and cuDNN**
   - Ensure all versions are compatible with each other.

3. **Verify Installations**
   - Check GPU drivers and CUDA availability using:
     ```sh
     nvidia-smi
     ```
   - Check CUDA toolkits with:
     ```sh
     nvcc --version
     ```

4. **Verify TensorFlow GPU Availability**
   - Install TensorFlow if not already installed:
     ```sh
     pip install tensorflow
     ```
   - Run the following Python code:
     ```python
     import tensorflow as tf
     print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
     ```

#### Virtual Environment Setup (Python 3.9.19)

1. **Install Virtualenv and Virtualenvwrapper**
   ```sh
   pip install virtualenv virtualenvwrapper

2. **Manipulate Virtualenv and Virtualenvwrapper**
```sh
mkvenv ml  # Create your environment
workon ml  # Activate your environment
deactivate # Deactivate your environment (when needed)
```

Once you are in your environment (`ml`), install the following using pip:

- matplotlib 3.7.5
- keras 2.9.0
- h5py 3.11.0
- numpy 1.24.3
- pandas 2.0.3
- scikit-learn 1.3.2
- scipy 1.10.1
- seaborn 0.13.2
- tensorflow 2.9.3

You will find the names of all the packages installed in our environment inside the repository (Environment Specs) folder!


### Model

#### Data
For data, we used the UrbanSound dataset: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html). 
A README file to understand how this dataset is structured is provided in the "Data" folder.

#### Data Preprocessing
For this part we use the makeinputdata_ps2024.py script which takes .wav formated sound files organized based on the urbansound dataset and we get 10 (classes) collections of spectrograms of the sounds

For each fold, the script performs the following steps:

1. **Setup Directory Path and Initialize Arrays**
   - Sets up the directory path for the fold.
   - Initializes arrays for spectrograms (`spgs`) and labels (`labels`).

2. **Load .wav Files**
   - Loads each `.wav` file using `librosa.load` with the specified sampling frequency.
   - Resizes the signal to the defined duration.

3. **Extract Labels**
   - Extracts the label from the filename of each audio file.

4. **Normalize the Signal**
   - Normalizes the signal to ensure consistent amplitude.

5. **Generate Power Spectrogram**
   - Iterates over time frames.
   - Applies a Blackman window to smooth the signal.
   - Computes the FFT (Fast Fourier Transform).
   - Creates a 2D array of power spectrogram values.

6. **Stack Arrays**
   - Stacks the spectrogram and label arrays to compile the data.

7. **Save Processed Data**
   - Saves the spectrograms and labels in a compressed `.npz` file for each fold.

These steps convert raw audio files into a standardized set of spectrograms, which can be used as input for machine learning models.


#### Model Conception
![Example Image](stagejapon/Model_analysis/Model_Architecture.png)

#### Model Training
For the training part we use the script learnmodel_ps2024.py.
The trining process includes:

1. **Set Parameters and Conditions**
   - Set parameters like `IMAGESIZE`, `SAMPLEFREQ`, `NNDROPOUT`, `EPOCHS`, and more.
   - Define the dataset source and frequency scale.

2. **One-Hot Encode Labels**
   - Define a function to one-hot encode the labels.

3. **Check Condition Parameters and Set Output File Names**
   - Set the dataset-specific parameters and check for the correct frequency scale.

4. **Setup Data Directory**
   - Set up the directory names for data and output.
   - Ensure the output directory exists or create it.

5. **Prepare Training and Test Data**
   - Split the dataset into training and test sets based on folds.
   - Load training data and labels, normalize, and apply transformations like Mel scaling if required.
   - Load and preprocess test data similarly.

6. **Build the Convolutional Neural Network**
   - Define the CNN architecture with convolutional layers, batch normalization, activation functions, max pooling, flattening, dense layers, and dropout.
   - Compile the model using the Adam optimizer and set the loss function and metrics.

7. **Train the Model**
   - Fit the model on the training data with specified batch size and epochs.
   - Monitor the training process using callbacks like early stopping, tensorboard, and model checkpoint.

8. **Evaluate the Model**
   - Predict on the test data and compute the confusion matrix.
   - Normalize the confusion matrix and print it.
   - Evaluate the model's performance on the test data and print the accuracy.

9. **Save the Model**
    - Save the trained CNN model and its parameters to a file in the output directory.

These steps train a convolutional neural network (CNN) to classify environmental sounds using the power spectrograms of the sounds.

#### Model Analysis
Discuss the results, including metrics and performance evaluation.

##### Confusion Matrix
##### Accuracy and loss


## Raspberry

We created a GUI for our model using Python (PyQt5 Libraririe) 



# **<span style="color:blue">Main Title in Bold and Blue</span>**

