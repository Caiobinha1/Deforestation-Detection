# Edge Machine Learning for Forest Acoustic Monitoring Demonstration

Este projeto consiste em um sistema de monitoramento acústico automatizado voltado para a segurança de florestas protegidas. O objetivo é identificar sons característicos de atividades de desmatamento ou intrusão, como motosserras e motores, utilizando redes neurais (YAMNet).

## Estrutura de Pastas

Para o funcionamento dos scripts, a estrutura de diretórios deve ser a seguinte:

- `Teste_de_Campo/Audios_Originais/`: Contém os áudios limpos gravados em campo (Modelos STIHL MS 170 e HT 75).
- `Teste_de_Campo/Forest_Ambiance/`: Contém os áudios de ruído ambiente (sons de floresta, pássaros, etc).
- `Teste_de_Campo/Generated_Audios/`: Pasta onde serão salvos os áudios misturados pelo script de SNR.

## Requisitos

- Python 3.x
- Bibliotecas: `numpy`, `soundfile`, `librosa`, `tensorflow`, `tensorflow-hub`

## Como utilizar

### 1. Geração de Dados Sintéticos (Mixagem)

O script `Mixaudios_SNR.py` é responsável por criar o dataset de teste. Ele combina os sinais das motosserras com o ruído ambiente em uma proporção específica de Signal-to-Noise Ratio (SNR).

- O padrão configurado é de 10 dB.
- Todos os arquivos são convertidos automaticamente para 16000 Hz (mono).
- Comando: `python Mixaudios_SNR.py`

### 2. Inferência em Lote

O script de classificação (ex: `BatchInference.py`) carrega o modelo treinado e processa a pasta de áudios gerados.

- O modelo utiliza a arquitetura YAMNet para extração de embeddings.
- Classes detectadas: `motosserra`, `motor`, `serra_manual` e `outros`.
- O sistema aplica um limiar (threshold) de 0.6 para considerar uma detecção como positiva para as classes de interesse.

## Detalhes Técnicos

- Taxa de amostragem alvo: 16000 Hz.
- Modelos de referência gravados: STIHL MS 170 e STIHL HT 75.
- A classe `outros` agrupa todos os sons ambientais que não são alvos de detecção do sistema de segurança.