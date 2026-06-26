# Bibliotecas
# Pytorch e Torchvision
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

# Uteis
import math
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import os
import random  
from PIL import Image
timer = __import__('timeit').default_timer
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
import gc

import time
from datetime import timedelta

import copy

import cv2
from PIL import Image

start_time = time.time()


def fixar_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Aplica as seeds globais imediatamente
fixar_seeds(42)

# Cria o gerador fixo para os DataLoaders
gerador_fixo = torch.Generator()
gerador_fixo.manual_seed(42)


print(torch.__version__)


if torch.cuda.is_available():
    device = "cuda"
    print(torch.version.cuda)
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device



ERRORSNAME = "../Rede_cnn_testes_finais/errors_incendio_intrus_1.png"


BATCH_SIZE = 500

IMG_HEIGHT = 320
IMG_WIDTH = 320

THRESHOLD = 0.5

VAL_FREQ_BATCHES = 50


USAR_EARLY_STOPPING = True  # Ligar/Desligar a paragem antecipada
PACIENCIA = 20              # Quantas validações sem melhoria vai percorrer antes de parar o treino 
MIN_DELTA = 0.001           # A loss tem de descer pelo menos isto para ser considerada melhoria


class_names =[
    "Nada",          # Índice 0
    "Fumo ou Fogo",  # Índice 1
    "People",        # Índice 2
    "Cat",           # Índice 3
    "Dog"            # Índice 4
]

# Número total de classes (será 4)
num_classes = 5

# Dicionário para converter o Índice (que sai do modelo) para Texto (para o print)
idx_to_label = {idx: name for idx, name in enumerate(class_names)}


# Transforms
transform_teste = transforms.v2.Compose([
    transforms.v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

transform = transforms.v2.Compose([
    transforms.v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
    
    transforms.v2.RandomHorizontalFlip(p=0.5), # 50% de chance de espelhar
    transforms.v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Rotação, movimento e zoom
    transforms.v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Variações de luz

    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

# Dataset personalizado
class CustomDataSet(Dataset):
    def __init__(self, csv_file, num_imagens, image_dir, tipo_dataset, transforms=None, offset=0):
        
        # Lê o CSV e força a coluna label a ser string
        labels_df = pl.read_csv(csv_file, separator=';', columns=["FILENAME", "label"], schema_overrides={"label": pl.String})
        if num_imagens is not None:
            labels_df = labels_df.slice(offset, num_imagens)

        self.image_dir = image_dir
        self.transforms = transforms
        self.filenames = labels_df["FILENAME"].to_list()
        raw_labels = labels_df["label"].to_list()

        self.targets = torch.zeros((len(raw_labels), 5), dtype=torch.float32)


        for idx, label_str in enumerate(raw_labels):
            label_str = str(label_str).strip() if label_str else ""
            
            # --- SE FOR IMAGEM DE INCÊNDIO ---
            if tipo_dataset == "incendios":
                # Tenta converter a string para inteiro de forma segura
                try:
                    label_original = int(label_str)
                except ValueError:
                    label_original = 3 # Se houver erro, assume "Nada"

                # A REGRA QUE PEDISTE:
                label = 0 if label_original == 3 else 1
                
                # Ativa o neurónio 0 ou 1
                self.targets[idx, label] = 1.0

            # --- SE FOR IMAGEM DE INTRUSOS ---
            elif tipo_dataset == "intrusos":
                # Se o CSV estiver vazio ou disser "None", consideramos que é "Nada" (Classe 0)
                if label_str == "" or label_str.lower() == "none":
                    self.targets[idx, 0] = 1.0
                else:
                    # Lê cada número da string (ex: "02" vira 0 e 2)
                    for char in label_str:
                        if char.isdigit():
                            label_original = int(char)
                            
                            #soma 2 para passar para 2, 3 e 4:
                            novo_label = label_original + 2
                            
                            self.targets[idx, novo_label] = 1.0
                                

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        complete_dir = os.path.join(self.image_dir, file_name)

        try:
            image = Image.open(complete_dir).convert("RGB")
            if self.transforms:
                image = self.transforms(image)
                
            target = self.targets[idx]
            return image, target
            
        except Exception as e:
            # Em caso de erro, devolve imagem preta para não crashar o treino
            print(f"Erro ao carregar a imagem {file_name}: {e}")
            return torch.zeros((3, IMG_HEIGHT, IMG_WIDTH)), self.targets[idx]

# Modelo
class RedeNeuronal(nn.Module):
    def __init__(self, num_classes):
        super(RedeNeuronal, self).__init__()

    
        self.conv_layer = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(10)
        self.max_pool = nn.AdaptiveMaxPool2d(10)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 10 * 10 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)  

        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)

        x = torch.cat([x_avg, x_max], dim=1)

        #x = self.max_pool(x)
        #x = self.avg_pool(x)

        x = self.classifier(x)     

        return x

def load_model(modelo=0):

    model = RedeNeuronal(num_classes).to(device)

    caminho = modelo
    try:
        model.load_state_dict(torch.load(caminho, weights_only=True, map_location=device))
        model.to(device)
    except Exception as e:
        print(f"{e}")
        exit()

    return model

def avaliar_imagem(modelo, image_path, device, class_names, transforms, csv_path=None, threshold=0.5):
    modelo.eval()

    if not os.path.exists(image_path):
        print(f"Erro: A imagem '{image_path}' não foi encontrada.")
        return

    img_original = Image.open(image_path).convert("RGB")
    img_tensor = transforms(img_original)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = modelo(img_tensor)
        probs = torch.sigmoid(outputs)[0]
        
    # Procurar TODAS as classes previstas que passem o threshold
    classes_previstas = []
    for i, prob in enumerate(probs):
        if prob.item() > threshold:
            classes_previstas.append(class_names[i])

    label_prevista_str = ", ".join(classes_previstas) if classes_previstas else "Nenhuma classe detetada"

    label_real_str = "Desconhecida"
    real_encontrada = False
    
    # Só tenta ler se um caminho para o CSV for fornecido e se o ficheiro existir
    if csv_path and os.path.exists(csv_path):
        try:
            # Forçar a leitura como string para não desaparecer o zero à esquerda
            df = pl.read_csv(csv_path, separator=';', schema_overrides={"label": pl.String})
            nome_ficheiro = os.path.basename(image_path)
            resultado = df.filter(pl.col("FILENAME") == nome_ficheiro)
            
            if not resultado.is_empty():
                label_bruta = resultado["label"][0]
                if label_bruta is not None:
                    txt_label = str(label_bruta).strip()
                    nomes_reais = []
                    # Converter "02" em ["People", "Dog"]
                    for char in txt_label:
                        if char.isdigit() and int(char) < len(class_names):
                            nomes_reais.append(class_names[int(char)])
                    
                    label_real_str = ", ".join(nomes_reais) if nomes_reais else "Nenhuma"
                    real_encontrada = True  # Ativamos a flag porque a label foi encontrada!
            else:
                pass
                
        except Exception as e:
            pass
    
    #Apresentar Resultados na Consola de forma dinâmica
    if real_encontrada:
        print(f"Label Real: {label_real_str}")
        
    print(f"Label Prevista: {label_prevista_str.upper()}")
    print("\nDistribuição de probabilidades:")
    for i, nome in enumerate(class_names):
        p = probs[i].item() * 100
        marcador = "==> DETETADO" if p > (threshold*100) else ""
        print(f"  - {nome}: {p:.2f}% {marcador}")

    # Mostrar a imagem
    plt.figure(figsize=(6, 6))
    plt.imshow(img_original)
    
    #Construir título da imagem dinamicamente
    if real_encontrada:
        titulo = f"Real: {label_real_str}\nPrevisto: {label_prevista_str}"
    else:
        titulo = f"Previsto: {label_prevista_str}"
        
    plt.title(titulo, fontsize=12, fontweight='bold', color='blue')
    plt.axis('off')
    plt.savefig(ERRORSNAME)
    plt.close('all')

def analisar_video(modelo, video_path, output_path, device, class_names, transforms, threshold=0.5, multilabel=True, mostrar_prob=True):
    modelo.eval()
    
    # Abrir o vídeo de entrada
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo '{video_path}'")
        return
    
    # Obter propriedades do vídeo original
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar o escritor de vídeo para o ficheiro de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec padrão para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"A iniciar análise do vídeo: '{video_path}'")
    print(f"Dimensões: {width}x{height} | FPS: {fps:.2f} | Total de frames: {total_frames}")
    
    # Usamos o tqdm para mostrar uma barra de progresso no terminal
    with torch.no_grad():
        with tqdm(total=total_frames, desc="A processar vídeo", ncols=100) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # Fim do vídeo
                
                # O OpenCV lê os frames em formato BGR. É necessário converter para RGB para o PyTorch
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Converter o array do frame para imagem PIL para aplicar os teus transforms de forma idêntica
                pil_img = Image.fromarray(frame_rgb)
                img_tensor = transforms(pil_img).unsqueeze(0).to(device)
                
                # Passar o frame pelo modelo
                outputs = modelo(img_tensor)
                
                # Processar previsões dependendo do tipo de problema
                if multilabel:
                    # Lógica do teu modelo de INTRUSÕES (Sigmoid)
                    probs = torch.sigmoid(outputs)[0]
                    previstos = []
                    for i, prob in enumerate(probs):
                        p = prob.item()
                        if p > threshold:
                            if mostrar_prob:
                                previstos.append(f"{class_names[i]} ({p*100:.1f}%)")
                            else:
                                previstos.append(class_names[i])
                    
                    if previstos:
                        texto_output = ", ".join(previstos)
                        cor_texto = (0, 255, 0) # Verde se houver intrusões ativas
                    else:
                        texto_output = "Nada detetado"
                        cor_texto = (0, 0, 255) # Vermelho se estiver limpo
                else:
                    # Lógica do teu modelo de INCÊNDIO/FUMO (Softmax)
                    probs = torch.softmax(outputs, dim=1)[0]
                    
                    # A probabilidade de Fumo ou Fogo está no índice 1 do teu modelo
                    prob_fogo = probs[1].item()
                    
                    # Se a probabilidade de Fogo for maior ou igual ao threshold
                    if prob_fogo >= threshold:
                        cor_texto = (0, 255, 0)  # Verde em BGR (Perigo/Chama ativa)
                        if mostrar_prob:
                            texto_output = f"Fumo ou Fogo ({prob_fogo*100:.1f}%)"
                        else:
                            texto_output = "Fumo ou Fogo"
                    else:
                        cor_texto = (0, 0, 255)  # Vermelho em BGR (Seguro/Sem chama)
                        if mostrar_prob:
                            texto_output = f"Fumo ou Fogo ({prob_fogo*100:.1f}%)"
                        else:
                            texto_output = "Fumo ou Fogo"
                
                # Desenhar o texto da previsão no frame original (em BGR)
                cv2.putText(
                    frame, 
                    texto_output, 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2,            # Tamanho da fonte
                    cor_texto,      
                    3,              # Espessura da linha
                    cv2.LINE_AA
                )
                
                # Escrever o frame anotado no novo ficheiro de vídeo
                out.write(frame)
                pbar.update(1)
                
    # Libertar os recursos da memória
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVídeo processado com sucesso! Guardado em: '{output_path}'")




modelo=load_model("código_deuc/codigos_finais/modelo_final_intrus_incendio.pt") 
# https://www.youtube.com/watch?v=whlymAuRtzU


video_input_path = "código_deuc/codigos_finais/video_teste.mp4"
video_output_path = "código_deuc/codigos_finais/video_teste_analisado.mp4"

analisar_video(modelo, video_input_path, video_output_path, device, class_names, transform_teste, threshold=0, multilabel=True, mostrar_prob=1)

#avaliar_imagem(modelo, "código_deuc/codigos_finais/image.png", device, class_names, transform_teste)
