"""Inference demo for YAMNet."""
from __future__ import division, print_function

import sys
import requests
import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
#import matplotlib.pyplot as plt
import params as yamnet_params
import yamnet as yamnet_model
#record audio libs
import pyaudio
import wave
import io

import spidev
# Message codes (you can define your own protocol)
SPI_MSG_SHOT = 0x53    # 'S' in ASCII
SPI_MSG_CHAINSAW = 0x43 # 'C' in ASCII
SPI_CHANNEL = 0
SPI_DEVICE = 0
SPI_SPEED_HZ = 500000

#telegram:
BOT_TOKEN = "7737724651:AAEjnMMOY6IP7UvEcmVTUy-0CXwEarUoJM4"
CHAT_ID = "7520514571"

def enviar_audio_telegram(audio_segment, sr, bot_token, chat_id):
    # Cria um buffer de bytes em memoria
    buffer = io.BytesIO()
    
    # Salva o audio no buffer (formato WAV)
    sf.write(buffer, audio_segment, sr, format='WAV')
    buffer.seek(0)  # Volta ao inicio do buffer para leitura
    
    # Configuracao do Telegram
    url = f"https://api.telegram.org/bot{bot_token}/sendAudio"
    files = {'audio': ('alerta.wav', buffer, 'audio/wav')}
    data = {'chat_id': chat_id, 'caption': 'Ameaca detectada!'}
    
    # Envia o audio
    response = requests.post(url, files=files, data=data)
    return response.json()
    
def send_spi_message(message_code):
    try:
        spi = spidev.SpiDev()
        spi.open(SPI_CHANNEL, SPI_DEVICE)
        spi.max_speed_hz = SPI_SPEED_HZ
        spi.mode = 0b00
        
        # Send the message code
        spi.xfer2([message_code])
        
        spi.close()
        print(f"SPI message sent: {hex(message_code)}")
    except Exception as e:
        print(f"Error sending SPI message: {e}")
        
# Audio configuration
FORMAT = pyaudio.paInt16  # 32-bit format
CHANNELS = 1              # Stereo
RATE = 16000              # Sampling rate (Hz)
CHUNK = 1024              # Buffer size
RECORD_SECONDS =15        # Duration
DEVICE_INDEX = 2          # Change if necessary
OUTPUT_FILENAME = "output.wav"


def process_detection(sound_name, class_index, scores, wav_data, sr, threshold, spi_message):
    """
    Helper function to process a detection: segments audio, sends Telegram alerts, and SPI message.
    """
    # Find the frame with the highest score for the target class
    detected_window_index = np.argmax(scores[:, class_index])
    max_score = scores[detected_window_index, class_index]

    print(f"Possible {sound_name} detected with confidence: {max_score:.2f}")
    
    # --- Audio Segmentation Logic ---
    x, y = 1, 1
    # Find the end of the event by looking for scores above a lower threshold
    while (detected_window_index + x) < len(scores) and scores[detected_window_index + x, class_index] >= 0.2:
        x += 1
    # Find the start of the event
    while (detected_window_index - y) > 0 and scores[detected_window_index - y, class_index] >= 0.2:
        y += 1
    
    # Each 'window' in scores represents about 0.48 seconds of audio (half the hop sec of params)
    hop_seconds = yamnet_params.Params().patch_hop_seconds
    start_time_sec = (detected_window_index - y) * hop_seconds
    end_time_sec = (detected_window_index + x) * hop_seconds
    
    # Extract the audio segment in samples
    start_sample = max(0, int(start_time_sec * sr))
    end_sample = min(len(wav_data), int(end_time_sec * sr))
    segment = wav_data[start_sample:end_sample]

    # --- Alerting Logic ---
    mensagem = f"Possible {sound_name} detected with max confidence: {max_score:.3f}"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={mensagem}"
    requests.get(url)
    enviar_audio_telegram(segment, sr, BOT_TOKEN, CHAT_ID)
    send_spi_message(spi_message)
    
    return True  # Indicate that a detection was made

def main(argv, limit, limit_shot):
  #  assert argv, 'Usage: inference.py <wav file> <wav file> ...'

  params = yamnet_params.Params()
  yamnet = yamnet_model.yamnet_frames_model(params)
  yamnet.load_weights('yamnet.h5')
  yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv').tolist()
  # =============================================================================
  # START OF IMPROVED DETECTION LOGIC
  # =============================================================================

  # 1. Define a configuration for all target sounds.
  # This makes your code scalable and easy to read.
  TARGET_SOUNDS = {
      'Chainsaw':         {'threshold': 0.5, 'spi_code': SPI_MSG_CHAINSAW},
      'Engine':           {'threshold': 0.5, 'spi_code': SPI_MSG_CHAINSAW},
      'Gunshot, gunfire': {'threshold': 0.5, 'spi_code': SPI_MSG_SHOT},
      'Screaming':        {'threshold': 0.5, 'spi_code': SPI_MSG_SHOT},
  }

  # 2. Get class indices dynamically by name (safer than hardcoding).
  for sound_name in list(TARGET_SOUNDS.keys()):
      try:
          class_index = yamnet_classes.index(sound_name)
          TARGET_SOUNDS[sound_name]['index'] = class_index
      except ValueError:
          print(f"Warning: Class name '{sound_name}' not found. It will be ignored.")
          del TARGET_SOUNDS[sound_name] # Remove if not found in class map
  
  # Decode the WAV file.
  wav_data, sr = sf.read(argv, dtype=np.int16)
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  waveform = waveform.astype('float32')

  # Convert to mono and the sample rate expected by YAMNet.
  if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)
  if sr != params.sample_rate:
    waveform = resampy.resample(waveform, sr, params.sample_rate)

  # Predict YAMNet classes.
  scores, embeddings, spectrogram = yamnet(waveform)
  scores = scores.numpy()

  detection_made = False
  # 3. Loop through the configuration to check for sounds.
  for sound_name, config in TARGET_SOUNDS.items():
      class_index = config['index']
      class_scores = scores[:, class_index] # Get all scores over time for this class
      
      # Check if the highest score for this sound exceeds its threshold
      if np.max(class_scores) >= config['threshold']:
          # If detected, call the helper function to handle alerts and segmentation
          process_detection(
              sound_name=sound_name,
              class_index=class_index,
              scores=scores,
              wav_data=wav_data,
              sr=sr,
              threshold=config['threshold'],
              spi_message=config['spi_code']
          )
          detection_made = True
          # Optional: break here if you only want to report the first/loudest sound found
          break 

  if not detection_made:
      send_spi_message(0x00) # Send null message if nothing was detected
      
  # =============================================================================
  # END OF IMPROVED DETECTION LOGIC
  # =============================================================================
      
  # This part for printing general top results can remain as it is
  prediction = np.mean(scores, axis=0)
  top5_i = np.argsort(prediction)[::-1][:5]
  print("maiores medias no audio:",argv, ':\n' +
        '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                  for i in top5_i))


if __name__ == '__main__':
  while True:
    limit = 0
    limit_shot = 0
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        input_device_index=DEVICE_INDEX,
                        frames_per_buffer=CHUNK)

    print("Gravando...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Gravacao completa.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Combine all chunks into a single byte string
    raw_audio = b''.join(frames)

    # Convert raw bytes to a NumPy array (int16)
    audio_np = np.frombuffer(raw_audio, dtype=np.int16)

    # Apply 200% gain (2x amplification)
    audio_np = (audio_np * 4.0).astype(np.int16)  # Multiply & convert back to int16

    # Clip to prevent overflow (if needed)
    audio_np = np.clip(audio_np, -32768, 32767)   # Ensure no clipping for int16

    # Convert back to bytes
    amplified_audio = audio_np.tobytes()

    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
      wf.setnchannels(CHANNELS)
      wf.setsampwidth(audio.get_sample_size(FORMAT))
      wf.setframerate(RATE)
      wf.writeframes(amplified_audio)

    print(f"Audio saved as {OUTPUT_FILENAME}")

    main(OUTPUT_FILENAME, limit, limit_shot)