# Project: Training CNN Model on Ubuntu with GPU and application using Raspberry GUI App

## Introduction
This project is developed as part of our internship research topic at OYAMA INSTITUTE OF TECHNOLOGY under the guidance of Dr. HIRATA Katsumi. It focuses on creating a GUI application for the Raspberry Pi 4 that uses a Convolutional Neural Network (CNN) model to classify environmental sounds based on their spectrograms. The project involves setting up the necessary software environment on Ubuntu 20.04.6 LTS, configuring GPU support, and developing the model and the GUI application.

## Table of Contents
1. [Machine](#machine)
   - [Installing Latest Version of Ubuntu (20.04.6 LTS)](#installing-latest-version-of-ubuntu-20046-lts)
   - [Configuring Wired Connection on Ubuntu (If You Don't Have WIFI)](#configuring-wired-connection-on-ubuntu-if-you-dont-have-wifi)
   - [Setup Environment for GPU Usage](#setup-environment-for-gpu-usage)
     - [Local Setup](#local-setup)
     - [Virtual Environment Setup (Python 3.8.10)](#virtual-environment-setup-python-3810)
   - [Model](#model)
     - [Data](#data)
     - [Data Preprocessing](#data-preprocessing)
     - [Model Conception](#model-conception)
     - [Model Training](#model-training)
     - [Model Analysis](#model-analysis)
2. [Raspberry](#raspberry)
   - [Using the GUI app](#using-the-gui-app)

## Installing Latest Version of Ubuntu (20.04.6 LTS)

## Installing Latest Version of Ubuntu (20.04.6 LTS)

To set up your machine for this project, you need to install the latest version of Ubuntu, which is 20.04.6 LTS (Focal Fossa). Follow the steps below:

1. Download Ubuntu 20.04.6 LTS (Focal Fossa)
   - Visit the official Ubuntu releases page: [Ubuntu 20.04.6 LTS](https://releases.ubuntu.com/focal/)
   - Download the ISO file from the provided link.

2. Create a Bootable USB
   - **For Windows Users:**
     - Download and install [Rufus](https://rufus.ie/).
     - Insert your USB drive and open Rufus.
     - Select the downloaded Ubuntu ISO file and the USB drive. 
     - Click "Start" to create the bootable USB.
   - **For Mac Users:**
     - Download and install [Balena Etcher](https://www.balena.io/etcher/).
     - Insert your USB drive and open Balena Etcher.
     - Select the downloaded Ubuntu ISO file and the USB drive.
     - Click "Flash" to create the bootable USB.

3. Install Ubuntu
   - Plug the USB drive into the target machine.
   - Start the machine and enter BIOS mode (this is typically done by pressing the "Del" key during startup, but it may vary depending on your system).
   - In the BIOS menu, set the primary boot device to the USB drive.
   - Save changes and exit the BIOS menu. The machine should now boot from the USB drive.
   - Follow the on-screen instructions to install Ubuntu:
     - Select your language and keyboard layout.
     - Choose "Install Ubuntu" and follow the prompts.
     - When asked about installation type, you can select "Erase disk and install Ubuntu" if this machine is dedicated to this project. **Note: This will delete all data on the machine's hard drive.**
     - Follow the rest of the prompts to complete the installation.

### Additional Steps
   - After the installation is complete, remove the USB drive and restart the machine.
   - Follow any additional on-screen prompts to complete the setup, including creating a user account and setting up your network connection.

By following these steps, you will have the latest version of Ubuntu installed and ready for the next stages of your project setup.


## Configuring Wired Connection on Ubuntu (If You Don't Have WIFI)

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

## Setup Environment for GPU Usage

### Local Setup

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

### Virtual Environment Setup (Python 3.8.10)

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

You will find all the packages installed in our environment inside the repository (Environment Specs) folder!


## Model

We used a simple model with the following structure:

- **Input Layer**
  - 728 neurons (inputs)

- **Hidden Layers**
  - 4 Hidden Layers, each containing:
    - Batch Normalization
    - Activation: using ReLU function
    - Max Pooling

- **Output Layer**
  - 10 classes, each corresponding to a type of sound.



## Data

For data, we used the UrbanSound dataset: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html). 

A README file to understand how this dataset is structured is provided in the "Data" folder.





