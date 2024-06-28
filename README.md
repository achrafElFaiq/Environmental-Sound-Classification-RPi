# Stage EL FAIQ Achraf / BENJELLOUN Saad OYAMA INSTITUTE OF TECHNOLOGY

## Installing Latest Version of Ubuntu (20.04.6 LTS)

1. **Download Ubuntu 20.04.6 LTS (Focal Fossa)**
   - From the official website: [Ubuntu 20.04.6 LTS](https://releases.ubuntu.com/focal/)

2. **Create a Bootable USB**
   - Use Rufus for Windows or Balena Etcher for Mac to make your USB bootable with the downloaded ISO file.

3. **Install Ubuntu**
   - Plug the USB into the target machine and start it in BIOS mode (typically by pressing the "Del" key during startup, but this can vary).
   - Set the primary boot device to the USB.
   - Follow the on-screen instructions to install the OS on your machine.

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





