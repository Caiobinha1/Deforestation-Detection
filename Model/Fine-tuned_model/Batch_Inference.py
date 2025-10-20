import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# --- 1. DEFINE MODEL AND CLASS INFO ---

# The list of classes your model was trained on.
# This order is CRUCIAL for the confusion matrix.
MY_CLASSES = ['chainsaw', 'footsteps', 'crackling_fire', 'rain', 'engine', 'hand_saw']

# Path to your saved custom model
CUSTOM_MODEL_PATH = r'Project\Fine-tuned_model\esc50_forest.h5'


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

def get_prediction(waveform, yamnet_model, custom_classifier):
    """Runs inference and returns the predicted class name."""
    scores, embeddings, spectrogram = yamnet_model(waveform)
    predictions = custom_classifier.predict(embeddings, verbose=0)
    mean_predictions = np.mean(predictions, axis=0)
    inferred_class_index = np.argmax(mean_predictions)
    inferred_class_name = MY_CLASSES[inferred_class_index]
    return inferred_class_name

def main():
    # --- 2. LOAD MODELS ONCE ---
    print("Loading YAMNet model...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    print("Loading custom classifier...")
    try:
        custom_classifier = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    except Exception as e:
        print(f"Error loading custom model from '{CUSTOM_MODEL_PATH}': {e}")
        print("Please ensure the model path is correct.")
        return

    # --- 3. GET EVALUATION FOLDER ---
    eval_folder = input("Enter the path to the main evaluation folder (containing class subfolders): ")
    if not os.path.isdir(eval_folder):
        print(f"Error: Directory not found at '{eval_folder}'")
        return

    # --- 4. RUN INFERENCE AND COLLECT RESULTS ---
    print("\nStarting evaluation...")
    y_true = []  # To store the actual labels
    y_pred = []  # To store the model's predictions

    # Iterate through subdirectories in the evaluation folder
    class_folders = [d for d in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, d))]

    for class_folder_name in class_folders:
        # Determine the true label from the folder name
        true_label = None
        for class_name in MY_CLASSES:
            if class_name.replace("_", "") in class_folder_name.replace("_", ""):
                true_label = class_name
                break
        
        if true_label is None:
            print(f"Warning: Skipping folder '{class_folder_name}' as its name does not match any known class.")
            continue

        print(f"\nProcessing folder for class: '{true_label}'")
        
        # Find all .wav files in the current class folder
        audio_files = glob.glob(os.path.join(eval_folder, class_folder_name, '*.wav'))
        if not audio_files:
            print(f"  No .wav files found in '{class_folder_name}'.")
            continue

        for audio_file in audio_files:
            waveform = load_wav_16k_mono(audio_file)
            if waveform is None:
                continue
            
            predicted_label = get_prediction(waveform, yamnet_model, custom_classifier)
            
            y_true.append(true_label)
            y_pred.append(predicted_label)
            
            print(f"  File: {os.path.basename(audio_file):<30} | True: {true_label:<15} | Predicted: {predicted_label:<15}")

    if not y_true:
        print("\nEvaluation complete, but no audio files were processed. Please check your folder structure.")
        return
        
    # --- 5. CALCULATE AND DISPLAY RESULTS ---
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=MY_CLASSES)
    
    print("Confusion Matrix (rows=True, cols=Predicted):")
    print(f"{'':<15}" + " ".join([f"{cls[:7]:<7}" for cls in MY_CLASSES]))
    for i, row in enumerate(cm):
        print(f"{MY_CLASSES[i]:<15}" + " ".join([f"{val:<7}" for val in row]))

    # Plot and save the confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=MY_CLASSES)
    display.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    output_filename = 'confusion_matrix.png'
    plt.savefig(output_filename)
    print(f"\nGraphical confusion matrix saved as '{output_filename}'")
    
    # plt.show() # Uncomment this line if you want the plot to pop up automatically

if __name__ == '__main__':
    main()