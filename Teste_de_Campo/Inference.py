import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

def load_and_preprocess_audio(file_path, target_sr=16000):
    """
    Carrega o áudio e garante que esteja em 16kHz e mono.
    """
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None

def calculate_grouped_means(model_path, audio_dir, class_names, threshold=0.6):
    """
    Roda a inferência, agrupa os resultados e calcula a taxa de detecção 
    com base em um threshold para as classes alvo.
    """
    print("Carregando o modelo YAMNet base do TensorFlow Hub para extrair embeddings...")
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)

    print(f"Carregando o modelo classificador de: {model_path}")
    classifier_model = tf.keras.models.load_model(model_path)
    
    valid_extensions = ('.wav', '.mp3', '.flac')
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(valid_extensions)]
    
    if not audio_files:
        print("Nenhum arquivo de áudio encontrado no diretório especificado.")
        return

    print(f"\nIniciando inferência em {len(audio_files)} arquivos...\n")
    
    grouped_predictions = {}
    
    # Variáveis para a funcionalidade do Threshold
    target_classes = ['motosserra', 'motor', 'serra_manual']
    total_audios = 0
    detected_audios = 0
    
    for current_idx, file_name in enumerate(audio_files, 1):
        file_path = os.path.join(audio_dir, file_name)
        audio_data = load_and_preprocess_audio(file_path)
        
        if audio_data is None or len(audio_data) == 0:
            continue
            
        # Extração e Classificação
        _, embeddings, _ = yamnet_model(audio_data)
        predictions = classifier_model.predict(embeddings, verbose=0)
        file_mean_prediction = np.mean(predictions, axis=0)
        
        # --- LÓGICA DO THRESHOLD INDIVIDUAL ---
        total_audios += 1
        predicted_idx = np.argmax(file_mean_prediction)
        
        if predicted_idx < len(class_names):
            predicted_class = class_names[predicted_idx]
            max_prob = file_mean_prediction[predicted_idx]
            
            # Se a classe prevista for um dos nossos alvos e a confiança passar do threshold
            if predicted_class in target_classes and max_prob >= threshold:
                detected_audios += 1
        
        # --- LÓGICA DO AGRUPAMENTO ---
        original_audio_name = file_name.split('_with_')[0]
        
        if original_audio_name not in grouped_predictions:
            grouped_predictions[original_audio_name] = []
        
        grouped_predictions[original_audio_name].append(file_mean_prediction)
        
        if current_idx % 10 == 0:
            print(f"Processado: {current_idx}/{len(audio_files)}")
            
    # --- EXIBIÇÃO AGRUPADA ---
    print("\n" + "=" * 110)
    print(f"{'Áudio Original':<25} | {'Prevista (Média)':<15} | {'Confiança Média':<15} | {'Probabilidades Detalhadas'}")
    print("-" * 110)
    
    for base_name, preds_list in grouped_predictions.items():
        mean_preds = np.mean(preds_list, axis=0)
        predicted_idx = np.argmax(mean_preds)
        
        if predicted_idx < len(class_names):
            predicted_class = class_names[predicted_idx]
        else:
            predicted_class = f"Desconhecida ({predicted_idx})"
            
        confidence = mean_preds[predicted_idx] * 100
        details = " | ".join([f"{c}: {m*100:.1f}%" for c, m in zip(class_names, mean_preds)])
        
        print(f"{base_name:<25} | {predicted_class:<15} | {confidence:>6.2f}%         | {details}")

    # --- EXIBIÇÃO DO RESUMO DE THRESHOLD ---
    print("\n" + "=" * 110)
    print("--- Resumo de Detecção do Sistema ---")
    print(f"Utilizando um threshold de {threshold} para as classes target {target_classes}, "
          f"seria detectado {detected_audios} de {total_audios} áudios totais.")
    print("=" * 110 + "\n")

if __name__ == "__main__":
    MODEL_PATH = 'Teste_de_Campo\esc50_forest_Demo.h5' 
    AUDIO_DIRECTORY = 'Teste_de_Campo\Generated_Audios'
    
    EXPECTED_CLASSES = ['motosserra', 'motor', 'serra_manual', 'outros']
    DETECTION_THRESHOLD = 0.5
    
    calculate_grouped_means(MODEL_PATH, AUDIO_DIRECTORY, EXPECTED_CLASSES, threshold=DETECTION_THRESHOLD)