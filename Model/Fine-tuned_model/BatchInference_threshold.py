import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
import pandas as pd # Using pandas for a nicely formatted output table

# --- 1. DEFINE MODEL AND CLASS INFO ---

# This order MUST EXACTLY MATCH the training notebook.
MY_CLASSES = ['chainsaw', 'footsteps', 'crackling_fire', 'rain', 'engine', 'hand_saw']

# Path to your saved custom model
CUSTOM_MODEL_PATH = r'Project\Fine-tuned_model\esc50_forest.h5'

# --- NEW: Set your desired threshold ---
# A value of 0.5 means the model must be at least 50% confident.
THRESHOLD = 0.22


def load_wav_16k_mono(filename):
    """
    Loads a WAV file, resamples to 16kHz, and truncates to the first 15 seconds.
    """
    try:
        wav_data, sr = sf.read(filename, dtype='float32')
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None
    
    if wav_data.ndim > 1:
        wav_data = np.mean(wav_data, axis=1)
        
    if sr != 16000:
        num_samples = round(len(wav_data) * 16000 / sr)
        wav_data = scipy.signal.resample(wav_data, num_samples)
        
    max_samples = 15 * 16000
    if len(wav_data) > max_samples:
        wav_data = wav_data[:max_samples]
        
    return wav_data.astype(np.float32)

def get_mean_probabilities(waveform, yamnet_model, custom_classifier):
    """Runs inference and returns the mean probabilities for ALL classes."""
    scores, embeddings, spectrogram = yamnet_model(waveform)
    predictions = custom_classifier.predict(embeddings, verbose=0)
    # The softmax function converts the model's raw output (logits) into probabilities
    probabilities = tf.nn.softmax(predictions, axis=-1).numpy()
    mean_probabilities = np.mean(probabilities, axis=0)
    return mean_probabilities

def main():
    # --- 2. LOAD MODELS ONCE ---
    print("Loading YAMNet model...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    print("Loading custom classifier...")
    try:
        custom_classifier = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    except Exception as e:
        print(f"Error loading custom model from '{CUSTOM_MODEL_PATH}': {e}")
        return

    # --- 3. GET EVALUATION FOLDER ---
    eval_folder = input("Enter the path to the main evaluation folder (containing class subfolders): ")
    if not os.path.isdir(eval_folder):
        print(f"Error: Directory not found at '{eval_folder}'")
        return

    # --- 4. RUN INFERENCE AND COLLECT RESULTS ---
    print(f"\nStarting evaluation with a threshold of {THRESHOLD}...")
    
    # Initialize a dictionary to store TP, FP, FN, TN for each class
    results = {class_name: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for class_name in MY_CLASSES}
    
    class_folders = [d for d in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, d))]

    for class_folder_name in class_folders:
        true_label = None
        for class_name in MY_CLASSES:
            if class_name.replace("_", "") in class_folder_name.replace("_", ""):
                true_label = class_name
                break
        
        if true_label is None:
            print(f"Warning: Skipping folder '{class_folder_name}' as its name does not match any known class.")
            continue

        print(f"\nProcessing folder for class: '{true_label}'")
        
        audio_files = glob.glob(os.path.join(eval_folder, class_folder_name, '*.wav'))
        if not audio_files:
            print(f"  No .wav files found in '{class_folder_name}'.")
            continue

        for audio_file in audio_files:
            waveform = load_wav_16k_mono(audio_file)
            if waveform is None:
                continue
            
            # Get the array of mean probabilities for all classes
            mean_probs = get_mean_probabilities(waveform, yamnet_model, custom_classifier)
            
            # --- NEW THRESHOLD LOGIC ---
            # Iterate through each class to update its TP, FP, FN, TN counts
            for i, current_class in enumerate(MY_CLASSES):
                prob_for_class = mean_probs[i]
                
                is_present = (current_class == true_label)
                is_detected = (prob_for_class >= THRESHOLD)

                if is_detected and is_present:
                    results[current_class]['TP'] += 1
                elif is_detected and not is_present:
                    results[current_class]['FP'] += 1
                elif not is_detected and is_present:
                    results[current_class]['FN'] += 1
                elif not is_detected and not is_present:
                    results[current_class]['TN'] += 1

    # --- 5. CALCULATE AND DISPLAY RESULTS ---
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS (Threshold = {THRESHOLD})")
    print("="*80)

    # Prepare data for a summary table
    summary_data = []

    for class_name in MY_CLASSES:
        tp = results[class_name]['TP']
        fp = results[class_name]['FP']
        fn = results[class_name]['FN']
        tn = results[class_name]['TN']

        # Calculate metrics, handling division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        summary_data.append({
            "Class": class_name,
            "Precision": f"{precision:.2%}",
            "Recall": f"{recall:.2%}",
            "F1-Score": f"{f1_score:.2f}",
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

        # Print the 2x2 confusion matrix for each class
        print(f"\n--- Confusion Matrix for: {class_name} ---")
        print(f"                  Predicted POSITIVE | Predicted NEGATIVE")
        print(f"Actual POSITIVE:        {tp:^10} |       {fn:^10}")
        print(f"Actual NEGATIVE:        {fp:^10} |       {tn:^10}")
        print("-" * 55)

    print("\n--- Overall Metrics Summary ---\n")
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("\n" + "="*80)

if __name__ == '__main__':
    main()