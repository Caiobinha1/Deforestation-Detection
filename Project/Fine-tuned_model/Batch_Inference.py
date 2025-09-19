import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import os
import glob

# --- 1. DEFINE MODEL AND CLASS INFO ---

# The list of classes your model was trained on
MY_CLASSES = ['chainsaw', 'footsteps', 'crackling_fire', 'rain', 'engine', 'hand_saw']

# Path to your saved custom model (use raw string or forward slashes)
CUSTOM_MODEL_PATH = r'Project\Fine-tuned_model\esc50_forest.h5'

def load_wav_16k_mono(filename):
    """ Loads a WAV file, converts it to a float tensor, and resamples to 16kHz. """
    try:
        wav_data, sr = sf.read(filename, dtype='float32')
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None
    
    # Ensure the audio is mono
    if wav_data.ndim > 1:
        wav_data = np.mean(wav_data, axis=1)
        
    # Resample if necessary
    if sr != 16000:
        num_samples = round(len(wav_data) * 16000 / sr)
        wav_data = scipy.signal.resample(wav_data, num_samples)
        
    return wav_data.astype(np.float32)

def main():
    # --- 2. LOAD BOTH MODELS (ONCE AT THE START) ---
    print("Loading YAMNet model...")
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)

    print("Loading custom classifier...")
    custom_classifier = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    
    # --- 3. GET FOLDER AND FIND AUDIO FILES ---
    audio_folder = input("Enter the path to the folder with audio files to classify: ")
    
    audio_files_to_classify = glob.glob(os.path.join(audio_folder, '*.wav'))
    
    if not audio_files_to_classify:
        print(f"No .wav files found in '{audio_folder}'. Please check the path.")
        return
        
    print(f"\nFound {len(audio_files_to_classify)} audio files to classify.")
    
    # --- 4. LOOP THROUGH EACH FILE AND RUN INFERENCE ---
    for audio_file in audio_files_to_classify:
        print("-" * 50)
        print(f"Processing: {os.path.basename(audio_file)}")
        
        waveform = load_wav_16k_mono(audio_file)
        if waveform is None:
            continue

        # Step 1: Extract embeddings with YAMNet
        scores, embeddings, spectrogram = yamnet_model(waveform)

        # Step 2: Classify embeddings with the custom model
        # `predictions` will have a shape of (num_windows, num_classes)
        predictions = custom_classifier.predict(embeddings, verbose=0)

        # --- 5. DISPLAY THE OVERALL RESULT (Main sound) ---
        mean_predictions = np.mean(predictions, axis=0)
        inferred_class_index = np.argmax(mean_predictions)
        inferred_class_name = MY_CLASSES[inferred_class_index]
        print(f"The main sound is: {inferred_class_name}")

        # --- 6. NEW: DISPLAY MAX VALUE FOR EACH CLASS PER WINDOW ---
        print("\n  Max prediction scores per class:")
        # Find the max score for each class across all windows (axis=0)
        max_scores_per_class = np.max(predictions, axis=0)
        # Find the window index where that max score occurred for each class
        max_score_indices = np.argmax(predictions, axis=0)
        
        for i, class_name in enumerate(MY_CLASSES):
            max_score = max_scores_per_class[i]
            window_index = max_score_indices[i]
            # Use formatting to align the output neatly
            print(f"    - {class_name:<15}: {max_score:.4f}")

    print("-" * 50)
    print("\nAll files have been classified.")


if __name__ == '__main__':
    main()