# ARARA: Monitoramento Acústico de Desmatamento Ilegal via Computação na Borda

Este repositório contém o código-fonte, scripts de pré-processamento de áudio, modelo pré-treinado e os dados de validação do trabalho **"ARARA: Monitoramento Acústico de Desmatamento Ilegal via Computação na Borda"**, submetido ao **XLIV Simpósio Brasileiro de Telecomunicações e Processamento de Sinais (SBrT 2026)**.

---

## Destaque dos Testes de Campo (SBrT 2026)

> *"Foi estruturado um repositório no GitHub contendo tanto os áudios originais capturados nestes testes de campo quanto versões desses misturados a ruídos de ambientes florestais com SNR de 10dB. Todos os arquivos deste conjunto foram classificados corretamente, permitindo verificar a eficácia do sistema mesmo em cenários de interferência acústica."*  
> — **Artigo ARARA (SBrT 2026)**

---

## Visão Geral da Arquitetura

O sistema **ARARA** (*Acoustic Recognition and Alerting for Remote Areas*) é uma solução de monitoramento florestal autônoma em tempo real projetada para operação remota.

- **Detecção na Borda (Hardware):** Utiliza um Raspberry Pi 3 acoplado ao kit de desenvolvimento LoRaWAN B-L072Z-LRWAN1, alimentado por um sistema solar (painel de 10 W, bateria de 12 V / 9 Ah e controlador PWM).
- **Modelo Neural:** Emprega a arquitetura **YAMNet** pré-treinada no AudioSet para extração de embeddings (1024 dimensões) a partir de sinais amostrados em 16 kHz mono, seguidos por uma camada de classificação customizada (512 neurônios ReLU + Softmax de 4 classes: `motosserra`, `motor`, `serra_manual` e `outros`).
- **Transmissão Eficiente:** Transmite apenas a classe identificada em um *payload* LoRaWAN compacto de 1 byte, reduzindo drasticamente o consumo energético do rádio e viabilizando a autonomia do dispositivo em ambientes remotos.

---

## Estrutura do Repositório

```
Deforestation-Detection/
├── README.md                          # Documentação acadêmica principal
├── requirements.txt                   # Dependências do projeto Python
├── .gitignore                         # Arquivos temporários ignorados pelo Git
├── LICENSE                            # Licença MIT
│
├── Teste_de_Campo/                    # Conjunto oficial de testes de campo (citado no artigo)
│   ├── Audios_Originais/              # Áudios gravados em campo (Motosserras STIHL MS 170 e HT 75)
│   ├── Forest_Ambiance/               # Ruídos de fundo florestais (chuva, pássaros, insetos)
│   ├── Generated_Audios/              # Áudios misturados sinteticamente em SNR de 10 dB
│   ├── esc50_forest_Demo.h5           # Modelo classificador refinado (YAMNet + Transfer Learning)
│   ├── Mixaudios_SNR.py               # Script de mixagem em lote sob SNR de 10 dB
│   └── BatchInference.py              # Script principal de inferência em lote
│
└── Analises_e_Resultados/            # Resultados estendidos e artefatos visuais
    ├── evaluate_extended.py           # Script de geração dos gráficos de desempenho
    ├── matriz_confusao_oficial.png    # Matriz de Confusão Normalizada Oficial
    ├── desempenho_motosserras_snr.png  # Confiança de detecção por modelo de motosserra (SNR 10dB)
    └── Relatorio_Inferencial_ARARA.pdf # Relatório técnico completo em PDF
```

---

## Requisitos e Instalação

### Pré-requisitos
- Python 3.9 ou superior
- Bibliotecas descritas no `requirements.txt`

### Instalação

```bash
git clone https://github.com/Caiobinha1/Deforestation-Detection.git
cd Deforestation-Detection
pip install -r requirements.txt
```

---

## Como Executar os Experimentos de Campo

### 1. Geração de Dados de Teste (Mixagem em Lote com SNR 10dB)

O script `Mixaudios_SNR.py` combina os sinais originais das motosserras gravadas em campo com ruídos florestais sob uma Relação Sinal-Ruído (SNR) de 10 dB, reamostrando todos os arquivos automaticamente para 16000 Hz mono:

```bash
python Teste_de_Campo/Mixaudios_SNR.py
```

*Os arquivos gerados são salvos automaticamente na pasta `Teste_de_Campo/Generated_Audios/`.*

### 2. Inferência em Lote e Classificação

O script `BatchInference.py` carrega a rede YAMNet base do TensorFlow Hub e o modelo classificador `esc50_forest_Demo.h5`, processando os áudios gerados e emitindo o relatório de detecção por agregação de média temporal:

```bash
python Teste_de_Campo/BatchInference.py
```

### 3. Geração dos Gráficos de Avaliação

Para visualizar e exportar a Matriz de Confusão e os gráficos de desempenho:

```bash
python Analises_e_Resultados/evaluate_extended.py
```

---

## Resultados da Validação de Campo

- **Acurácia de Detecção no Teste de Campo:** 100% de classificação correta dos eventos de motosserra sob interferência florestal (SNR 10 dB).
- **Tempo Médio de Inferência:** 1,25 segundos por segmento de áudio no Raspberry Pi 3.
- **Latência Fim-a-Fim:** 30 segundos (incluindo tempo de despertar por energia, aquisição de áudio de 10s, inferência local e transmissão LoRaWAN).

---

## Citação (BibTeX)

Se você utilizar este repositório ou os dados em sua pesquisa, por favor cite o trabalho:

```bibtex
@inproceedings{arara_sbrt2026,
  author    = {Caio M. Carlos and Leonardo C. C. Bitencourt and Walter A. Gontijo and Eduardo L. O. Batista and Richard D. Souza},
  title     = {ARARA: Monitoramento Acústico de Desmatamento Ilegal via Computação na Borda},
  booktitle = {Anais do XLIV Simpósio Brasileiro de Telecomunicações e Processamento de Sinais (SBrT 2026)},
  address   = {Salvador, BA},
  pages     = {1--5},
  year      = {2026},
  month     = {sep}
}
```

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE) - consulte o arquivo para detalhes.