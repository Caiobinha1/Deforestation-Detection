import librosa
import soundfile as sf
import numpy as np
import os
import glob

# --- Configuration ---
TARGET_SR = 16000
DURATION_S = 15 # You can adjust the target duration here

def adjust_audio_length(audio_array, target_length):
    """A helper function to truncate or pad an array to a target length."""
    current_length = len(audio_array)
    if current_length > target_length:
        return audio_array[:target_length]
    elif current_length < target_length:
        return np.pad(audio_array, (0, target_length - current_length))
    return audio_array

def main():
    """Main function to run the audio mixing process."""
    # --- 1. Get folder paths from the user ---
    folder1 = input("Enter the path to the first audio folder (e.g., chainsaws): ")
    folder2 = input("Enter the path to the second audio folder (e.g., forests): ")
    output_folder = input("Enter the path to the output folder for mixed audio: ")

    # --- 2. Find all .wav files in the input folders ---
    # The pattern '*.wav' finds all files ending with .wav
    files1 = glob.glob(os.path.join(folder1, '*.wav'))
    files2 = glob.glob(os.path.join(folder2, '*.wav'))

    if not files1 or not files2:
        print("Error: No .wav files found in one or both directories. Please check the paths.")
        return

    # --- 3. Create the output directory if it doesn't exist ---
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nFound {len(files1)} files in the first folder and {len(files2)} in the second.")
    print("Starting the mixing process...\n")

    num_samples = int(DURATION_S * TARGET_SR)

    # --- 4. Loop through every combination of files ---
    for file1_path in files1:
        # Load the first audio file ONCE for the inner loop
        sound1, _ = librosa.load(file1_path, sr=TARGET_SR)
        sound1 = adjust_audio_length(sound1, num_samples)
        
        # Get the base name for file naming (e.g., "Chainsaw1")
        base_name1 = os.path.splitext(os.path.basename(file1_path))[0]

        for file2_path in files2:
            # Get the base name for the second file (e.g., "Forest_ambiance3")
            base_name2 = os.path.splitext(os.path.basename(file2_path))[0]
            
            print(f"-> Mixing '{base_name1}' with '{base_name2}'...")

            # Load the second audio file
            sound2, _ = librosa.load(file2_path, sr=TARGET_SR)
            sound2 = adjust_audio_length(sound2, num_samples)

            # Mix the audio by adding the arrays
            combined_sound = sound1 + sound2

            # Prevent Clipping
            max_amplitude = np.max(np.abs(combined_sound))
            if max_amplitude > 1.0:
                combined_sound /= max_amplitude

            # --- 5. Generate the output filename and save ---
            output_filename = f"{base_name1}_{base_name2}.wav"
            output_path = os.path.join(output_folder, output_filename)
            
            sf.write(output_path, combined_sound, TARGET_SR)

    print(f"\n All done! Mixed files are saved in: {output_folder}")

# --- Run the main function when the script is executed ---
if __name__ == "__main__":
    main()