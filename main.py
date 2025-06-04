import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import queue
import sys
import threading
from resemblyzer import VoiceEncoder, preprocess_wav
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtCore import QTimer, QTimerEvent

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5  # сек
SIMILARITY_THRESHOLD = 0.7
SILENCE_TIMEOUT = 15  # сек

q_audio = queue.Queue()
lamp_on = False
reference_embedding = None
import socket

def send_dmx_255(level, channel):
    packet = bytearray(18 + 512)
    packet[0:8] = b'Art-Net\x00'
    packet[8] = 0x00  # OpCode low byte (ArtDMX)
    packet[9] = 0x50  # OpCode high byte
    packet[10] = 0x00
    packet[11] = 14
    packet[12] = 0x00
    packet[13] = 0x00
    packet[14] = 0x00  # Subnet/Universe
    packet[15] = 0x00
    packet[16] = 0x02  # Length Hi
    packet[17] = 0x00  # Length Lo
    packet[18 + channel - 1] = level   # Значение канала 1 = 255


    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(packet, ('255.255.255.255', 6454))

class LampWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.color = QColor(50, 50, 50)
        self.setWindowTitle("Лампа")
        self.setGeometry(100, 100, 240, 260)
        self.timer = QTimer()
        self.timer.timeout.connect(self.repaint)
        self.timer.start(100)
        from PyQt5.QtWidgets import QPushButton
        self.button = QPushButton("Записать эталон", self)
        self.button.setGeometry(60, 180, 120, 30)
        self.button.clicked.connect(self.record_reference)
        from PyQt5.QtWidgets import QCheckBox
        self.artnet_checkbox = QCheckBox("Включить ArtNet", self)
        self.artnet_checkbox.setChecked(True)
        self.artnet_checkbox.setGeometry(60, 20, 120, 20)
        self.setStyleSheet("QPushButton { font-size: 12px; } QCheckBox { font-size: 12px; }")

    def paintEvent(self, e):
        qp = QPainter(self)
        qp.setBrush(self.color)
        qp.drawEllipse(70, 60, 100, 100)

    def turn_on(self):
        self.color = QColor(0, 255, 0)

    def turn_off(self):
        self.color = QColor(50, 50, 50)

    def record_reference(self):
        from PyQt5.QtCore import QTimer
        self.button.setText("Идёт запись...")
        self.button.setEnabled(False)
        self.button.setStyleSheet("background-color: red")
        print("Запись эталона...")

        def finish_recording():
            sf.write("reference.wav", self._recorded_audio, SAMPLE_RATE)
            print("Эталон записан.")
            self.button.setStyleSheet("background-color: green")
            self.button.setText("Записано")

            global reference_embedding
            reference_embedding = VoiceEncoder().embed_utterance(preprocess_wav("reference.wav"))

            def reset_button():
                self.button.setText("Записать эталон")
                self.button.setStyleSheet("")
                self.button.setEnabled(True)

            QTimer.singleShot(2000, reset_button)

        duration = 10
        self._recorded_audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=CHANNELS)
        QTimer.singleShot(duration * 1000, finish_recording)

def audio_callback(indata, frames, time_info, status):
    q_audio.put(indata.copy())

def save_wav(data, filename):
    sf.write(filename, data, SAMPLE_RATE)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_similar(audio_data, ref_emb):
    encoder = VoiceEncoder()
    wav = preprocess_wav(audio_data)
    emb = encoder.embed_utterance(wav)
    sim = cosine_similarity(emb, ref_emb)
    print("Similarity:", sim)
    return sim >= SIMILARITY_THRESHOLD

def recorder_thread():
    global lamp_on
    buffer = []
    encoder = VoiceEncoder()

    last_match_time = 0
    while True:
        data = q_audio.get()
        buffer.append(data)

        if len(buffer) * BLOCK_DURATION >= 2.5:
            audio = np.concatenate(buffer, axis=0)
            sf.write("temp.wav", audio, SAMPLE_RATE)
            if is_similar("temp.wav", reference_embedding):
                last_match_time = time.time()
                if not lamp_on:
                    print("Голос совпал — включаем лампу")
                    if widget.artnet_checkbox.isChecked():
                        send_dmx_255(255, 345)
                    app.postEvent(widget, QTimerEvent(0))
                    widget.turn_on()
                    lamp_on = True
            elif lamp_on and time.time() - last_match_time > SILENCE_TIMEOUT:
                print("Нет совпадения 5 секунд — выключаем лампу")
                if widget.artnet_checkbox.isChecked():
                    send_dmx_255(0, 345)
                app.postEvent(widget, QTimerEvent(0))
                widget.turn_off()
                lamp_on = False
            buffer = []





if __name__ == "__main__":
    print("Загрузка эталона...")
    reference_embedding = VoiceEncoder().embed_utterance(preprocess_wav("reference.wav"))
    if reference_embedding is None or np.linalg.norm(reference_embedding) == 0:
        print("Ошибка: эталон пустой")
        sys.exit(1)

    app = QApplication(sys.argv)
    widget = LampWidget()
    widget.show()

    stream = sd.InputStream(callback=audio_callback, channels=CHANNELS,
                            samplerate=SAMPLE_RATE,
                            blocksize=int(SAMPLE_RATE * BLOCK_DURATION))
    stream.start()

    threading.Thread(target=recorder_thread, daemon=True).start()

    sys.exit(app.exec_())