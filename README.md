# Sistema Integrado para Deteção de Incêndios e Intrusões com Base em Abordagens de Deep Learning

Sistema de visão computacional baseado em Redes Neuronais Convolucionais (CNNs) para deteção em tempo real de incêndios (fumo/fogo) e intrusões (pessoas, gatos e cães) a partir de imagens. O projeto explora tanto modelos especializados como uma arquitetura unificada capaz de detetar ambos os tipos de ameaça num único pipeline de inferência.

> Projeto apresentada à Universidade de Aveiro para obtenção do grau de Licenciatura em Engenharia Computacional.

---

## 📁 Estrutura do Projeto

```
ficheiros_entrega/
├── código_modelos
│   ├── incendio_deuc_final.py
│   ├── intrus_deuc_final.py
│   ├── intrus_incendio_deuc_final_teste_video.py
│   └── intrus_incendio_deuc_final.py
├── códigos_dados
│   ├── count_by_dataset.py
│   ├── csv_maker_pastas.py
│   ├── CSVmaker_intrusoes.py
│   ├── CSVmaker.py
│   ├── CSVmixer.py
│   ├── Juntar_dataset.py
│   ├── remover_datasets.py
│   └── teste_download_dataset.py
└── csv
    ├── dataset_train_shuffled.csv
    ├── dataset_val.csv
    ├── labels_final_test.csv
    ├── labels_final_train_shuffled.csv
    └── labels_final_val.csv
```

### `código_modelos/`
Scripts responsáveis pela definição, treino e avaliação das arquiteturas CNN:

| Ficheiro | Descrição |
|---|---|
| `incendio_deuc_final.py` | Modelo binário final de deteção de fumo/fogo. |
| `intrus_deuc_final.py` | Modelo final de deteção de intrusões. |
| `intrus_incendio_deuc_final.py` | Modelo unificado, capaz de detetar incêndios e intrusões simultaneamente através de um tronco convolucional partilhado. |
| `intrus_incendio_deuc_final_teste_video.py` | Script de teste qualitativo do modelo unificado sobre um vídeo externo. |

### `códigos_dados/`
Scripts auxiliares de construção, limpeza e organização dos datasets:

| Ficheiro | Descrição |
|---|---|
| `teste_download_dataset.py` | Download seletivo de imagens (Open Images V7, COCO) via FiftyOne. |
| `CSVmaker.py` | Geração dos ficheiros CSV de etiquetas a partir das imagens recolhidas para o dataset de fumo/fogo. |
| `CSVmaker_intrusoes.py` | Geração de CSVs de etiquetas específicas para o dataset de intrusões. |
| `csv_maker_pastas.py` | Geração de CSVs com base na estrutura de pastas dos datasets de origem. |
| `CSVmixer.py` | Combinação e baralhamento (shuffle) dos CSVs de diferentes datasets/origens. |
| `Juntar_dataset.py` | Agregação dos vários datasets públicos num único conjunto. |
| `count_by_dataset.py` | Contagem e análise da distribuição de imagens/labels por dataset e por classe. |
| `remover_datasets.py` | Limpeza/remoção de imagens ou registos indesejados dos datasets. |

### `csv/`
Ficheiros de metadados com os nomes das imagens e respetivas etiquetas, usados pelos `Dataset` do PyTorch:

| Ficheiro | Descrição |
|---|---|
| `dataset_train_shuffled.csv` | Conjunto de treino baralhado de fumo/fogo. |
| `dataset_val.csv` | Conjunto de validação de fumo/fogo. |
| `labels_final_train_shuffled.csv` | Etiquetas finais de treino, após consolidação dos datasets de intrusões. |
| `labels_final_val.csv` | Etiquetas finais de validação de intrusões. |
| `labels_final_test.csv` | Etiquetas finais de teste de intrusões. |

---

## 🧠 Resumo do Sistema

- **Deteção de Incêndios** — modelo binário treinado sobre o dataset *FireSmokeDS* (~140.000 imagens), com **94.91% de accuracy** e **95.64% de F1-Score**.
- **Deteção de Intrusões** — modelo multi-classe com perda `BCEWithLogitsLoss`, treinado sobre ~118.365 imagens agregadas de COCO, Open Images e Objects365, atingindo **79.20% de accuracy**.
- **Modelo Unificado** — partilha as camadas convolucionais de extração de características entre os dois domínios, reduzindo significativamente a latência de inferência face à execução sequencial de dois modelos especializados (**~3.05 ms/imagem** vs. ~7.5 ms/imagem combinados).

---

## 💾 Datasets e Modelos Treinados

Devido ao volume de dados e ao tamanho dos checkpoints treinados, estes ficheiros **não estão incluídos neste repositório**. Estão disponíveis para download através do FCCN FileSender:

- 📦 **Datasets** (imagens + CSVs de etiquetas):
  [Download](https://filesender.fccn.pt/?s=download&token=85313828-0bb7-4fd6-b25d-1c8112643cd8)

- 🧠 **Modelos finais treinados** (checkpoints `.pt`):
  [Download](https://filesender.fccn.pt/?s=download&token=dde39e82-da0e-4663-ba8d-5f13d6d5f27b)
  > Os modelos finais não foram incluídos no GitHub por serem demasiado pesados para o repositório.

> ⚠️/ Os links do FCCN FileSender estão disponíveis até 25/8/2026
---/

## 👤 Autor

**Alexandre Santos Sousa**
Licenciatura em Engenharia Computacional — Universidade de Aveiro

**Orientação científica:**
- Prof. Doutor Carlos Couto — Departamento de Engenharia Civil, Universidade de Aveiro
- Mestre Rúben Santo — Universidade de Aveiro
