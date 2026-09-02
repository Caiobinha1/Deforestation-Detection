import os
import random
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def calculate_rms(audio: np.ndarray) -> float:
    """Calcula o Root Mean Square (RMS) de um sinal de áudio."""
    return np.sqrt(np.mean(audio ** 2))

def mix_audio_with_snr(signal_path: Path, noise_path: Path, output_path: Path, target_snr_db: float, target_sr: int = 16000) -> None:
    """
    Mistura um arquivo de sinal (classe alvo) com um arquivo de ruído ambiente
    ajustando a relação sinal-ruído (SNR em dB) e reamostrando para a taxa alvo.
    """
    try:
        # Carregamento e reamostragem em 16kHz mono (padrão YAMNet)
        signal, _ = librosa.load(signal_path, sr=target_sr, mono=True)
        noise, _ = librosa.load(noise_path, sr=target_sr, mono=True)

        signal_len = len(signal)
        noise_len = len(noise)

        if signal_len > noise_len:
            repeats = int(np.ceil(signal_len / noise_len))
            noise = np.tile(noise, repeats)[:signal_len]
        elif signal_len < noise_len:
            start = random.randint(0, noise_len - signal_len)
            noise = noise[start:start + signal_len]
        
        rms_signal = calculate_rms(signal)
        rms_noise = calculate_rms(noise)

        if rms_noise < 1e-10 or rms_signal < 1e-10:
            print(f"[Aviso] Áudio silencioso detectado. Ignorando mixagem de {signal_path.name}.")
            return

        snr_linear = 10 ** (target_snr_db / 20.0)
        noise_scaling_factor = rms_signal / (rms_noise * snr_linear)
        
        adjusted_noise = noise * noise_scaling_factor
        mixed_audio = signal + adjusted_noise
        
        # Normalização de amplitude
        max_amplitude = np.max(np.abs(mixed_audio))
        if max_amplitude > 1.0:
            mixed_audio /= max_amplitude
            
        sf.write(output_path, mixed_audio, target_sr)

    except Exception as e:
        print(f"[Erro] Falha ao processar {signal_path.name} com {noise_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Script de Mixagem de Áudio em Lote com SNR para Testes de Campo (ARARA)")
    parser.add_argument("--signal_dir", type=str, default=None, help="Caminho para a pasta com os áudios originais")
    parser.add_argument("--noise_dir", type=str, default=None, help="Caminho para a pasta com ruídos florestais")
    parser.add_argument("--output_dir", type=str, default=None, help="Caminho para salvar os áudios misturados")
    parser.add_argument("--snr", type=float, default=10.0, help="Relação Sinal-Ruído alvo em dB (padrão: 10.0)")
    parser.add_argument("--sr", type=int, default=16000, help="Taxa de amostragem alvo em Hz (padrão: 16000)")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    signal_dir = Path(args.signal_dir) if args.signal_dir else base_dir / "Audios_Originais"
    noise_dir = Path(args.noise_dir) if args.noise_dir else base_dir / "Forest_Ambiance"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "Generated_Audios"

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = ('.wav', '.flac', '.mp3')
    target_files = [f for f in signal_dir.glob("*") if f.suffix.lower() in valid_exts]
    noise_files = [f for f in noise_dir.glob("*") if f.suffix.lower() in valid_exts]

    if not target_files or not noise_files:
        print("[Erro] Diretórios de entrada vazios ou não encontrados.")
        return

    total_mixes = len(target_files) * len(noise_files)
    print(f"Iniciando mixagem em lote ({total_mixes} combinações) sob SNR de {args.snr} dB a {args.sr} Hz...")

    current_mix = 0
    for target_file in target_files:
        for noise_file in noise_files:
            current_mix += 1
            output_filename = f"{target_file.stem}_with_{noise_file.stem}_snr{int(args.snr)}db.wav"
            output_path = output_dir / output_filename
            
            mix_audio_with_snr(target_file, noise_file, output_path, args.snr, target_sr=args.sr)

    print(f"Mixagem concluída com sucesso! {total_mixes} arquivos gerados em: {output_dir}")

if __name__ == "__main__":
    main()