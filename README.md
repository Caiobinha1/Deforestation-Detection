# Deforestation Sound Detection Project

This project provides a complete pipeline for creating a custom audio dataset and classifying environmental sounds related to deforestation. It is organized into two main parts: a dataset validation and generation suite, and a fine-tuned machine learning model for inference.

---

## 📂 Project Structure

The repository is organized into two main folders, each with a specific purpose.
.
├── Dataset_validation/
│   ├── Mix_Audios.py               # Script to generate new audio clips
│   ├── Raw_Audios/                 # Folder for all source audio clips
│   │   ├── Chainsaw/
│   │   ├── Engine/
│   │   ├── Fire/
│   │   ├── Footsteps/
│   │   ├── Forest_Ambiance/
│   │   ├── HandSaw/
│   │   ├── Rain/
│   │   └── Thunderstorm/
│   └── Generated_Audios/           # Output folder for mixed audio
│
├── Fine-tuned_Model/
│   ├── [Your_Notebook_Name].ipynb    # Jupyter Notebook for model training
│   ├── batch_inference.py          # Script to classify a folder of audio
│   └── esc50_forest.h5             # The trained Keras model file
│
├── requirements.txt                # List of Python dependencies
└── README.md                       # This file
---
Workflow and Usage

Generate the Dataset:
The first step is to create a realistic dataset by mixing foreground sounds (e.g., a chainsaw) with background sounds (e.g., a forest ambiance).

Run the Script: Execute the mixing script from the project's root directory.

Follow Prompts: The script will ask you for the paths to the two source folders and a path for the output. The newly created audio files will be saved in Dataset_validation/Generated_Audios/.

Understand the Model (Optional)
If you want to see how the model was trained and fine-tuned:

Open and run the Jupyter Notebook located at Fine-tuned_Model/yamnet-transfer-learning-on-esc50.ipynb.

The final, trained model used by the inference script is Fine-tuned_Model/esc50_forest.h5.

Step 3: Classify Audio
Use the trained model to classify the sounds in your generated dataset or any other audio folder.

Run the Script: Execute the batch inference script from the project's root directory.
