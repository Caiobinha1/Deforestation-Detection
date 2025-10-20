# 🌲 Edge Machine Learning for Forest Acoustic Monitoring Implementation

This project is an end-to-end system designed to detect sounds associated with illegal deforestation in real-time. It uses a Raspberry Pi with a microphone to capture audio, a machine learning model to classify sounds, and a LoRaWAN end-node to send alerts over a long-range, low-power network.

The primary goal is to identify sounds such as **chainsaws, engines, hand saws, and footsteps** while ignoring natural ambient forest sounds.

## ✨ Features

- **Real-Time Audio Monitoring:** Continuously records audio from the environment in segments.
- **Machine Learning Inference:** Uses a fine-tuned YAMNet model (based on TensorFlow) to classify audio segments.
- **Targeted Event Detection:** Implements a threshold-based logic to specifically detect deforestation-related sounds.
- **Low-Power Alerting:** Communicates detections via SPI from the Raspberry Pi to a dedicated LoRaWAN board.
- **Long-Range Communication:** The STM32 LoRaWAN board transmits alerts to a remote gateway and server.

## 🛠️ Hardware & Software Requirements

### Hardware
- **Processing Unit:** Raspberry Pi (e.g., Raspberry Pi 4 Model B)
- **Audio Input:** USB Microphone
- **LoRaWAN Node:** STMicroelectronics B-L072Z-LRWAN1 Discovery kit

### Software
- **Raspberry Pi:**
  - Python 3.9+
  - TensorFlow / Keras
  - `pyaudio` for audio recording
  - `soundfile` & `scipy` for audio processing
  - `spidev` for SPI communication
  - `pandas` & `numpy`
- **STM32 LoRaWAN Node:**
  - STM32CubeIDE
  - B-L072Z-LRWAN1 Firmware Package

---

## 📁 Project Structure

The repository is organized into three main folders:

```
.
├── Model/
│   ├── Generate Dataset for Validation
│   ├── esc50_forest.h5
│   ├── Batch_Inference.py
│   └── Generated_Audios/
│
├── Raspberry/
│   ├── main.py
│   └── spi_test.py
│
├── Lorawan_end_node/
│   └── (STM32CubeIDE Project Files)
│
└── README.md
```

- **`/Model`**: Contains the machine learning components.
  - `esc50_forest.h5`: The trained and fine-tuned Keras model for sound classification.
  - `Batch_Inference.py`: Python script to evaluate the model's performance on a validation dataset.
  - `Generated_Audios/`: A dataset of mixed audio files (target sounds + noise) used for validation.

- **`/Raspberry`**: Contains the scripts that run on the Raspberry Pi.
  - `main.py`: The main application script that records audio, runs inference, and sends SPI alerts.
  - `spi_test.py`: A utility script to test the SPI communication between the Raspberry Pi and the STM32 board.

- **`/Lorawan_end_node`**: Contains the complete firmware project for the STM32 LoRaWAN board.
  - This folder should be opened with STM32CubeIDE to compile and flash the firmware onto the B-L072Z-LRWAN1 board.

---

## ⚙️ Setup & Installation

### 1. Raspberry Pi Setup

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/Caiobinha1/Deforestation-Detection/tree/main/Raspberry>
    cd <Raspberry>
    ```

2.  **Install Python Dependencies:** It is highly recommended to use a virtual environment.
    ```bash
    sudo apt-get update
    sudo apt-get install python3-pip python3-venv portaudio19-dev
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```


3.  **Enable SPI Interface:**
    - Run `sudo raspi-config`.
    - Navigate to `Interface Options` -> `SPI`.
    - Select `<Yes>` to enable the SPI interface.
    - Reboot the Raspberry Pi.

### 2. STM32 LoRaWAN Node Setup

1.  Open **STM32CubeIDE**.
2.  Go to `File > Open Projects from File System...` and select the `Lorawan_end_node` directory.
3.  Configure the LoRaWAN settings (e.g., AppEUI, DevEUI, AppKey) within the project as required by your LoRaWAN network server (e.g., The Things Network).
4.  Build the project and flash the firmware to the B-L072Z-LRWAN1 board.

### 3. Hardware Connection

1.  Connect the **USB Microphone** to one of the Raspberry Pi's USB ports.
2.  Connect the Raspberry Pi to the STM32 board via their **SPI pins**. Ensure the following connections are made:
    - RPi **SCLK** to STM32 **SPI_SCK**
    - RPi **MOSI** to STM32 **SPI_MOSI**
    - RPi **MISO** to STM32 **SPI_MISO**
    - RPi **CE0/CS0** to STM32 **SPI_NSS**
    - RPi **GND** to STM32 **GND**

---

## 🚀 Usage

1.  **Test the SPI Connection (Optional but Recommended):**
    - Navigate to the `Raspberry/` directory.
    - Run `python spi_test.py` to send a test byte and confirm the hardware connection is working.

2.  **Run the Main Detection Application:**
    - Ensure the STM32 board is powered on and running its firmware.
    - On the Raspberry Pi, navigate to the `Raspberry/` directory.
    - Make sure your virtual environment is active (`source .venv/bin/activate`).
    - Execute the main script:
      ```bash
      python main.py
      ```
    - The application will start recording audio in cycles. When a target sound is detected above the configured threshold, it will print an alert to the console and send the corresponding code via SPI to the LoRaWAN node.

## 📈 System Workflow

[USB Mic] ➡️ [**Raspberry Pi**] ➡️ [Audio Recording (`pyaudio`)] ➡️ [**ML Model Inference (`main.py`)**] ➡️ [Detection?] YES ➡️ [**SPI Alert**] ➡️ [**STM32 LoRaWAN Node**] ➡️ 📡 ➡️ [LoRaWAN Gateway]