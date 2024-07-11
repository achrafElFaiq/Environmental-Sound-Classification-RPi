import sys
import subprocess
import soundfile as sf
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QProgressBar
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
import numpy as np
import librosa
import math
import scipy as sp
import tensorflow as tf

class ModernFileChooserApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Minimalistic File Chooser')
        self.showFullScreen()
        self.setStyleSheet("background-color: #F9FAFB;")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        self.main_layout = QVBoxLayout()

        # Prediction result label (replaces welcome text)
        self.result_label = QLabel('Welcome back, dear friend!', self)
        self.result_label.setFont(QFont('Arial', 24, QFont.Bold))
        self.result_label.setStyleSheet("color: #333333;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.result_label)

        # Image label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)

        # Choose File button
        choose_file_button = QPushButton('Choose File', self)
        choose_file_button.setFont(QFont('Arial', 14))
        choose_file_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        choose_file_button.clicked.connect(self.showFileDialog)
        button_layout.addWidget(choose_file_button)

        # Record button
        self.record_button = QPushButton('Record', self)
        self.record_button.setFont(QFont('Arial', 14))
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #FF5733;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        self.record_button.clicked.connect(self.startRecording)
        button_layout.addWidget(self.record_button)

        # Recognize button
        recognize_button = QPushButton('Recognize', self)
        recognize_button.setFont(QFont('Arial', 14))
        recognize_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: #1E88E5;
            }
        """)
        recognize_button.clicked.connect(self.recognizeSound)
        button_layout.addWidget(recognize_button)

        # Exit button
        exit_button = QPushButton('Exit', self)
        exit_button.setFont(QFont('Arial', 14))
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: #B71C1C;
            }
        """)
        exit_button.clicked.connect(self.close)
        button_layout.addWidget(exit_button)

        self.main_layout.addLayout(button_layout)

        # Selected file label
        self.file_label = QLabel('No file selected', self)
        self.file_label.setFont(QFont('Arial', 12))
        self.file_label.setStyleSheet("color: #666666;")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.file_label)

        self.setLayout(self.main_layout)

    def showFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose Audio File", "", "Audio Files (*.wav);;All Files (*)",
                                                  options=options)
        if fileName:
            self.file_label.setText(f'Selected File: {fileName}')
            self.selected_file = fileName

    def startRecording(self):
        self.record_button.setEnabled(False)  # Disable the record button
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #FF5733;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #FF5733;
                width: 10px;
            }
        """)
        self.main_layout.addWidget(self.progress_bar)

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateProgressBar)
        self.timer.start(40)  # 4 seconds recording, so update every 40ms

        self.record_thread = threading.Thread(target=self.recordAudio)
        self.record_thread.start()

    def updateProgressBar(self):
        value = self.progress_bar.value()
        if value < 100:
            self.progress_bar.setValue(value + 1)
        else:
            self.timer.stop()
            self.main_layout.removeWidget(self.progress_bar)
            self.progress_bar.deleteLater()
            self.record_button.setEnabled(True)  # Re-enable the record button

    def recordAudio(self):
        command = ['arecord', '-d', '4', '-f', 'cd', 'output.wav']
        try:
            subprocess.run(command, check=True)
            self.selected_file = 'output.wav'
            self.recognizeSound()
        except subprocess.CalledProcessError as e:
            print(f"Error during recording: {e}")

    def printAudioProperties(self, filename):
        try:
            with sf.SoundFile(filename) as f:
                channels = f.channels
                sample_rate = f.samplerate
                frames = len(f)
                duration = frames / float(sample_rate)
                print(f"Audio Properties of {filename}:")
                print(f"Channels: {channels}")
                print(f"Sample Rate: {sample_rate} Hz")
                print(f"Number of Frames: {frames}")
                print(f"Duration: {duration:.2f} seconds")
        except Exception as e:
            print(f"Error reading audio file properties: {e}")

    def recognizeSound(self):
        if hasattr(self, 'selected_file'):
            inputfile = self.selected_file
            print('Sound file: ' + inputfile)

            IMAGESIZE = 128  # 128 / 256
            SAMPLEFREQ = 8192  # 8192 / 16384 / 32768
            DURATION = 4  # Duration in seconds
            twin = sp.signal.windows.blackman(IMAGESIZE * 4)
            source = 'u'  # 'u' for urbansound8k / 'e' for ESC50
            ftype = 'l'  # 'l' for Linear-scale / 'm' for Mel-scale

            # Setting data parameters based on the source
            if source == 'u':
                sounddata = 'urbansound'
                numclass = 10
                numfold = 10
            elif source == 'e':
                sounddata = 'esc50'
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
                print('!! Invalid input type !!')
                sys.exit(1)

            sig, s = librosa.load(inputfile, sr=SAMPLEFREQ, mono=True)
            sig = np.resize(sig, SAMPLEFREQ * DURATION)

            # Normalize
            sig = sig / np.max(np.abs(sig))

            step = math.floor((DATALENGTH - IMAGESIZE * 4) / (IMAGESIZE - 1))
            spg = np.zeros(IMAGESIZE)
            for timeframe in range(IMAGESIZE):
                subsig = sig[step * timeframe:step * timeframe + IMAGESIZE * 4] * twin
                tmpps = np.square(np.abs(np.fft.fft(np.asarray(subsig))))
                spg = np.vstack([spg, tmpps[0:IMAGESIZE]])

            spg = spg[1:, :]
            spg = spg.T  # Transpose

            # Mel scaling
            if ftype == 'm':
                spg[:, :] = librosa.feature.melspectrogram(S=spg[:, :], sr=4096)

            spg = spg / spg.max()  # Normalize
            spg[spg < 1e-8] = 1e-8  # Setting lower bound (0.00000001) to prevent log-of-zero
            spg = 10 * np.log10(spg)  # Decibelize
            spg = spg + 80  # Offset to make all values positive
            spg = spg[np.newaxis, :, :, np.newaxis]

            # Load CNN model and fitted parameters
            model_path = "/home/achrafandsaad/Desktop/Project-INSA/v1.h5"
            model = tf.keras.models.load_model(model_path)

            pred_out = model.predict(spg)
            pred_label = np.argmax(pred_out, axis=1)[0]

            classes = [
                "an air conditioner", "a car horn", "children playing", "a dog barking", "a drilling sound",
                "an engine idling", "a gun shot", "a jackhammer", "a siren", "street music"
            ]
            predicted_class = classes[pred_label]
            self.result_label.setFont(QFont('Arial', 24, QFont.Bold))
            self.result_label.setText(f'This appears to be {predicted_class}')

            # Display the corresponding image
            image_path = f"/home/achrafandsaad/Desktop/Project-INSA/Images/{pred_label}.png"  # Use .webp images
            print(pred_label)
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        else:
            print("No file selected for recognition")

def main():
    app = QApplication(sys.argv)
    ex = ModernFileChooserApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
