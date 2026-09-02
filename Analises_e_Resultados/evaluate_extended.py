import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Configuração de Caminhos Relativos Multiplataforma
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "Teste_de_Campo"
MODEL_PATH = TEST_DIR / "esc50_forest_Demo.h5"
ORIGINALS_DIR = TEST_DIR / "Audios_Originais"
AMBIANCE_DIR = TEST_DIR / "Forest_Ambiance"
GENERATED_DIR = TEST_DIR / "Generated_Audios"
OUTPUT_DIR = BASE_DIR / "Analises_e_Resultados"

CLASS_NAMES = ['motosserra', 'motor', 'serra_manual', 'outros']

def load_audio(file_path, target_sr=16000):
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None

def map_8_to_4_classes(probs_8):
    p0 = probs_8[0]
    p1 = probs_8[1]
    p2 = probs_8[2]
    p3 = np.sum(probs_8[3:])
    return np.array([p0, p1, p2, p3])

def run_evaluation():
    print("Carregando YAMNet e modelo refinado ARARA...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    classifier_model = tf.keras.models.load_model(MODEL_PATH)

    dataset_items = []
    
    # 1. Audios Originais (Target: motosserra -> index 0)
    for f in ORIGINALS_DIR.glob("*.wav"):
        dataset_items.append({'file_path': str(f), 'file_name': f.name, 'category': 'Original', 'true_idx': 0})

    # 2. Forest Ambiance (Target: outros -> index 3)
    for f in AMBIANCE_DIR.glob("*.wav"):
        dataset_items.append({'file_path': str(f), 'file_name': f.name, 'category': 'Ambiente', 'true_idx': 3})

    # 3. Generated Audios (Target: motosserra -> index 0)
    for f in GENERATED_DIR.glob("*.wav"):
        dataset_items.append({'file_path': str(f), 'file_name': f.name, 'category': 'Mistura SNR 10dB', 'true_idx': 0})

    results = []

    for idx, item in enumerate(dataset_items, 1):
        audio = load_audio(item['file_path'])
        if audio is None or len(audio) == 0:
            continue
        
        _, embeddings, _ = yamnet_model(audio)
        raw_preds_8 = classifier_model.predict(embeddings, verbose=0)
        preds_4 = np.array([map_8_to_4_classes(p) for p in raw_preds_8])
        
        mean_preds = np.mean(preds_4, axis=0)
        pred_idx = np.argmax(mean_preds)
        
        results.append({
            'file_name': item['file_name'],
            'category': item['category'],
            'true_idx': item['true_idx'],
            'pred_idx': pred_idx,
            'mean_preds': mean_preds
        })

    # Geração dos Gráficos Oficiais
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    y_true = [r['true_idx'] for r in results]
    y_pred = [r['pred_idx'] for r in results]
    
    # Matriz de Confusão Oficial
    plot_official_cm(y_true, y_pred, OUTPUT_DIR / "matriz_confusao_oficial.png")
    
    # Desempenho por Modelo de Motosserra em SNR 10dB
    plot_chainsaw_snr_performance(results, OUTPUT_DIR / "desempenho_motosserras_snr.png")

    print(f"Avaliação concluída! Resultados salvos em: {OUTPUT_DIR}")

def plot_official_cm(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            cm_norm[i] = cm[i] / row_sums[i]

    plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    ax.set_title("Matriz de Confusão Normalizada - Sistema ARARA (SBrT 2026)", fontsize=11, fontweight='bold', pad=12)
    ax.set_ylabel('Classe Real (Ground Truth)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Classe Predita na Borda', fontsize=10, fontweight='bold')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            norm_val = cm_norm[i, j]
            if count > 0 or row_sums[i] > 0:
                text_str = f"{norm_val:.2f}\n({count})"
            else:
                text_str = "-"
            ax.text(j, i, text_str,
                    ha="center", va="center",
                    color="white" if norm_val > 0.5 else "black",
                    fontsize=9, fontweight='bold')

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_chainsaw_snr_performance(results, save_path):
    gen_results = [r for r in results if r['category'] == 'Mistura SNR 10dB']
    chainsaw_groups = {}
    for r in gen_results:
        orig_name = r['file_name'].split('_with_')[0]
        if orig_name not in chainsaw_groups:
            chainsaw_groups[orig_name] = []
        chainsaw_groups[orig_name].append(r['mean_preds'][0] * 100)

    names = list(chainsaw_groups.keys())
    means = [np.mean(chainsaw_groups[n]) for n in names]
    stds = [np.std(chainsaw_groups[n]) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.bar(names, means, yerr=stds, capsize=5, color='#2b5c8f', edgecolor='black', alpha=0.85)
    ax.set_ylabel('Confiança Média em Motosserra (%)', fontsize=11, fontweight='bold')
    ax.set_title('Desempenho de Detecção por Gravação de Motosserra (SNR 10dB)', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=25, ha='right', fontsize=9, fontweight='bold')

    for bar, m in zip(bars, means):
        ax.annotate(f'{m:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, m),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    run_evaluation()
