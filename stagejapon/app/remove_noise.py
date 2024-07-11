import numpy as np
from scipy.io import wavfile
import noisereduce as nr
import argparse

def remove_noise(input_wav, output_wav):
    # Read the input wav file
    rate, data = wavfile.read(input_wav)

    # If the audio has more than one channel (stereo), we will convert it to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    # Write the output wav file
    wavfile.write(output_wav, rate, reduced_noise.astype(np.int16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove noise from a WAV file")
    parser.add_argument("input_wav", type=str, help="Path to the input WAV file")
    parser.add_argument("output_wav", type=str, help="Path to the output WAV file")

    args = parser.parse_args()

    remove_noise(args.input_wav, args.output_wav)

