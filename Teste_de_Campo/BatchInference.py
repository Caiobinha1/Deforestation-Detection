import os
import sys

# Silencia avisos verbosos de inicialização do TensorFlow no console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

# Suporte a UTF-8 no Windows terminal
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CLASS_NAMES = ['motosserra', 'motor', 'serra_manual', 'outros']
TARGET_CLASSES = ['motosserra', 'motor', 'serra_manual']

def load_and_preprocess_audio(file_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Carrega o áudio e converte para 16kHz mono (requisito da arquitetura YAMNet)."""
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"[Erro] Não foi possível carregar {file_path.name}: {e}")
        return None

def map_8_to_4_classes(probs_8: np.ndarray) -> np.ndarray:
    """Mapeia os 8 neurônios de saída para as 4 classes conceituais do artigo."""
    p0 = probs_8[0]
    p1 = probs_8[1]
    p2 = probs_8[2]
    p3 = np.sum(probs_8[3:])
    return np.array([p0, p1, p2, p3])

def run_batch_inference(model_path: Path, audio_dir: Path, threshold: float = 0.5):
    """
    Executa a inferência em lote utilizando a arquitetura YAMNet + Transfer Learning,
    agrupando os resultados por áudio original e calculando a taxa de detecção.
    """
    print("Carregando modelo YAMNet base (TensorFlow Hub)...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    print(f"Carregando classificador customizado ARARA: {model_path.name}")
    classifier_model = tf.keras.models.load_model(model_path)

    valid_extensions = ('.wav', '.mp3', '.flac')
    audio_files = [f for f in audio_dir.glob("*") if f.suffix.lower() in valid_extensions]

    if not audio_files:
        print(f"[Aviso] Nenhum arquivo de áudio encontrado em: {audio_dir}")
        return

    print(f"\nIniciando inferência em lote ({len(audio_files)} arquivos)...\n")

    grouped_predictions = {}
    total_audios = 0
    detected_audios = 0

    for current_idx, file_path in enumerate(audio_files, 1):
        audio_data = load_and_preprocess_audio(file_path)
        if audio_data is None or len(audio_data) == 0:
            continue

        _, embeddings, _ = yamnet_model(audio_data)
        raw_preds_8 = classifier_model.predict(embeddings, verbose=0)
        preds_4 = np.array([map_8_to_4_classes(p) for p in raw_preds_8])

        file_mean_prediction = np.mean(preds_4, axis=0)

        total_audios += 1
        predicted_idx = np.argmax(file_mean_prediction)
        predicted_class = CLASS_NAMES[predicted_idx]
        max_prob = file_mean_prediction[predicted_idx]

        if predicted_class in TARGET_CLASSES and max_prob >= threshold:
            detected_audios += 1

        original_audio_name = file_path.name.split('_with_')[0]
        if original_audio_name not in grouped_predictions:
            grouped_predictions[original_audio_name] = []

        grouped_predictions[original_audio_name].append(file_mean_prediction)

        if current_idx % 10 == 0 or current_idx == len(audio_files):
            print(f"Processado: {current_idx}/{len(audio_files)}")

    # Exibição da Tabela de Resultados
    print("\n" + "=" * 110)
    print(f"{'Áudio Original':<25} | {'Classe Predita':<15} | {'Confiança Média':<15} | {'Probabilidades Detalhadas'}")
    print("-" * 110)

    for base_name, preds_list in grouped_predictions.items():
        mean_preds = np.mean(preds_list, axis=0)
        predicted_idx = np.argmax(mean_preds)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = mean_preds[predicted_idx] * 100

        details = " | ".join([f"{c}: {m * 100:.1f}%" for c, m in zip(CLASS_NAMES, mean_preds)])
        print(f"{base_name:<25} | {predicted_class:<15} | {confidence:>6.2f}%         | {details}")

    print("=" * 110)
    print("--- Resumo de Detecção do Sistema ARARA ---")
    print(f"Utilizando limiar de decisão τ = {threshold} para as classes alvo {TARGET_CLASSES}:")
    print(f"Resultado: {detected_audios} de {total_audios} áudios detectados com sucesso ({detected_audios / total_audios * 100:.1f}% de taxa de acerto).")
    print("=" * 110 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Script de Inferência em Lote do Sistema ARARA (SBrT 2026)")
    parser.add_argument("--model", type=str, default=None, help="Caminho para o arquivo do modelo .h5")
    parser.add_argument("--audio_dir", type=str, default=None, help="Caminho para a pasta com os áudios de teste")
    parser.add_argument("--threshold", type=float, default=0.5, help="Limiar de decisão de confiança (padrão: 0.5)")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    model_path = Path(args.model) if args.model else base_dir / "esc50_forest_Demo.h5"
    audio_dir = Path(args.audio_dir) if args.audio_dir else base_dir / "Generated_Audios"

    run_batch_inference(model_path, audio_dir, threshold=args.threshold)

if __name__ == "__main__":
    main()
