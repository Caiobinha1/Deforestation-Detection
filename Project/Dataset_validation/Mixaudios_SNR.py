import numpy as np
import soundfile as sf
import os
import random
import librosa # Import the librosa library

def calculate_rms(audio):
    """Calculates the Root Mean Square of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def mix_audio_with_snr(signal_path, noise_path, output_path, target_snr_db, target_sr=16000):
    """
    Mixes a signal and noise audio file to a target SNR, with resampling.

    Args:
        signal_path (str): Path to the target class audio file.
        noise_path (str): Path to the ambiance noise file.
        output_path (str): Path to save the mixed audio file.
        target_snr_db (float): The desired Signal-to-Noise Ratio in dB.
        target_sr (int): The target sample rate for resampling.
    """
    try:
        # --- Resampling Step ---
        # Load and resample the audio files to the target sample rate
        # librosa.load automatically converts to mono, which is good practice here.
        signal, _ = librosa.load(signal_path, sr=target_sr, mono=True)
        noise, _ = librosa.load(noise_path, sr=target_sr, mono=True)

        # Handle different audio lengths
        signal_len = len(signal)
        noise_len = len(noise)

        if signal_len > noise_len:
            repeats = int(np.ceil(signal_len / noise_len))
            noise = np.tile(noise, repeats)[:signal_len]
        elif signal_len < noise_len:
            start = random.randint(0, noise_len - signal_len)
            noise = noise[start:start + signal_len]
        
        # --- Core SNR Logic ---
        rms_signal = calculate_rms(signal)
        rms_noise = calculate_rms(noise)

        if rms_noise < 1e-10 or rms_signal < 1e-10:
            print(f"Warning: One of the audio files is silent. Skipping mix for {os.path.basename(signal_path)}.")
            return

        snr_linear = 10**(target_snr_db / 20.0)
        noise_scaling_factor = rms_signal / (rms_noise * snr_linear)
        
        adjusted_noise = noise * noise_scaling_factor
        mixed_audio = signal + adjusted_noise
        
        # --- Normalization ---
        max_amplitude = np.max(np.abs(mixed_audio))
        if max_amplitude > 1.0:
            mixed_audio /= max_amplitude
            
        # Write the output file using the target sample rate
        sf.write(output_path, mixed_audio, target_sr)

    except Exception as e:
        print(f"Error processing {signal_path} and {noise_path}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    TARGET_CLASSES_DIR = 'Dataset_validation\Raw_Audios\HandSaw'
    AMBIANCE_NOISES_DIR = 'Dataset_validation\Raw_Audios\Forest_Ambiance'
    OUTPUT_DIR = 'Dataset_validation\Generated_Audios'
    TARGET_SNR_DB = 10.0
    TARGET_SR = 16000 # Set the desired sample rate for all files

    # --- Script Execution ---
    print(f"Starting audio mixing process (resampling all to {TARGET_SR} Hz)...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        target_files = [f for f in os.listdir(TARGET_CLASSES_DIR) if f.endswith(('.wav', '.flac', '.mp3'))]
        noise_files = [f for f in os.listdir(AMBIANCE_NOISES_DIR) if f.endswith(('.wav', '.flac', '.mp3'))]
    except FileNotFoundError as e:
        print(f"Error: Directory not found - {e}. Please create the '{TARGET_CLASSES_DIR}' and '{AMBIANCE_NOISES_DIR}' folders.")
        exit()

    if not target_files or not noise_files:
        print("Error: Input directories are empty. Please add audio files.")
        exit()

    total_mixes = len(target_files) * len(noise_files)
    current_mix = 0

    for target_file in target_files:
        for noise_file in noise_files:
            current_mix += 1
            print(f"Processing mix {current_mix}/{total_mixes}...")

            signal_path = os.path.join(TARGET_CLASSES_DIR, target_file)
            noise_path = os.path.join(AMBIANCE_NOISES_DIR, noise_file)
            
            target_name = os.path.splitext(target_file)[0]
            noise_name = os.path.splitext(noise_file)[0]
            output_filename = f"{target_name}_with_{noise_name}_snr{int(TARGET_SNR_DB)}db.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            mix_audio_with_snr(signal_path, noise_path, output_path, TARGET_SNR_DB, target_sr=TARGET_SR)
            
    print(f"\nMixing complete! {total_mixes} files were generated in the '{OUTPUT_DIR}' folder.")