import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import pyaudio
import wave
import spidev # <-- Added for SPI communication

# --- 1. DEFINE MODEL AND CLASS INFO ---

MY_CLASSES = ['chainsaw', 'footsteps', 'crackling_fire', 'rain', 'engine', 'hand_saw']
CUSTOM_MODEL_PATH = 'esc50_forest.h5'

# --- NEW: SPI & ALERTING CONFIGURATION ---
SPI_MSG_CHAINSAW = 0x43 # Hex for 'C'
SPI_MSG_ENGINE = 0x14

SPI_CHANNEL = 0
SPI_DEVICE = 0
SPI_SPEED_HZ = 500000

TARGET_SOUNDS = {
    'chainsaw': {'threshold': 0.8, 'spi_code': SPI_MSG_CHAINSAW},
    'engine':   {'threshold': 0.8, 'spi_code': SPI_MSG_ENGINE}
}

# --- AUDIO RECORDING CONFIGURATION ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
DEVICE_INDEX = 2
OUTPUT_FILENAME = "live_recording.wav"

# --- 2. LOAD BOTH MODELS ---
print("Loading YAMNet model...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("Loading custom classifier...")
custom_classifier = tf.keras.models.load_model(CUSTOM_MODEL_PATH)

# --- 3. HELPER FUNCTIONS ---

def send_spi_message(message_code):
    try:
        spi = spidev.SpiDev()
        spi.open(SPI_CHANNEL, SPI_DEVICE)
        spi.max_speed_hz = SPI_SPEED_HZ
        spi.mode = 0b00
        spi.xfer2([message_code])
        spi.close()
        print(f"SPI message sent: {hex(message_code)}")
    except Exception as e:
        print(f"Error sending SPI message: {e}")

def load_wav_16k_mono(filename):
    wav_data, sr = sf.read(filename, dtype='float32')
    if wav_data.ndim > 1:
        wav_data = np.mean(wav_data, axis=1)
    if sr != 16000:
        num_samples = round(len(wav_data) * 16000 / sr)
        wav_data = scipy.signal.resample(wav_data, num_samples)
    return wav_data.astype(np.float32)

# --- MAIN RECORDING AND INFERENCE LOOP ---
while True:
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=DEVICE_INDEX,
                        frames_per_buffer=CHUNK)
    print("\n" + "="*30)
    print(f"Recording for {RECORD_SECONDS} seconds...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        frames.append(stream.read(CHUNK))
    print("Recording complete.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    raw_audio = b''.join(frames)
    audio_np = np.frombuffer(raw_audio, dtype=np.int16)
    amplified_audio_np = audio_np * 2.0
    clipped_audio_np = np.clip(amplified_audio_np, -32768, 32767)
    amplified_audio_bytes = clipped_audio_np.astype(np.int16).tobytes()

    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(amplified_audio_bytes)
    print(f"Audio saved with 2x gain as {OUTPUT_FILENAME}")
    
    print(f"Loading and processing audio file: {OUTPUT_FILENAME}")
    waveform = load_wav_16k_mono(OUTPUT_FILENAME)

    print("Step 1: Extracting embeddings with YAMNet...")
    _, embeddings, _ = yamnet_model(waveform)

    print("Step 2: Classifying embeddings with the custom model...")
    predictions = custom_classifier.predict(embeddings, verbose=0)
    
    # --- MODIFIED: THRESHOLD-BASED ALERTING LOGIC USING MEAN SCORE ---
    detection_made = False
    
    # First, calculate the mean scores for all classes across the whole clip
    mean_scores = np.mean(predictions, axis=0)

    for sound_name, config in TARGET_SOUNDS.items():
        try:
            class_index = MY_CLASSES.index(sound_name)
            
            # Get the mean score for this specific class
            mean_score = mean_scores[class_index]
            
            # Check if the mean score exceeds the threshold
            if mean_score >= config['threshold']:
                print(f"!! ALERT: '{sound_name}' detected with MEAN confidence {mean_score:.2f} !!")
                send_spi_message(config['spi_code'])
                detection_made = True
                break # Stop checking after the first confirmed detection
        except ValueError:
            print(f"Warning: Class '{sound_name}' not found in MY_CLASSES. Skipping.")
            
    if not detection_made:
        send_spi_message(0x00)

    # --- 5. DISPLAY THE OVERALL RESULT (FOR INFORMATION) ---
    inferred_class_index = np.argmax(mean_scores)
    inferred_class_name = MY_CLASSES[inferred_class_index]
    confidence_score = np.max(mean_scores)

    print("-" * 30)
    print(f"Overall main sound is: {inferred_class_name.upper()} (Avg Confidence: {confidence_score:.2f})")
    print("-" * 30)