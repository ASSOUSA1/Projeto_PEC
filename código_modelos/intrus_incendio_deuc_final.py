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


# Caminhos para dados
TRAIN_DATA_DIR_INTRUS = "../Dataset_intrus_final/train/images/images"
TRAIN_DATA_LABELS_INTRUS = "../labels_final_train.csv"
VAL_DATA_DIR_INTRUS = "../Dataset_intrus_final/validation/images/images"
VAL_DATA_LABEL_INTRUS = "../labels_final_val.csv"
TEST_DATA_DIR_INTRUS = "../Dataset_intrus_final/test/images/images"
TEST_DATA_LABEL_INTRUS = "../labels_final_test.csv"

TRAIN_DATA_DIR_INCENDIO = "../FireSmokeDS_upsampled_resized_aug/train/images"
TRAIN_DATA_LABELS_INCENDIO = "../dataset_train_shuffled.csv"
TEST_DATA_DIR_INCENDIO = "../FireSmokeDS_upsampled_resized_aug/val/images"
TEST_DATA_LABEL_INCENDIO = "../dataset_val.csv"


os.makedirs("../Rede_cnn_testes_finais", exist_ok=True)

LOSSIMAGENAME = "../Rede_cnn_testes_finais/evolution_loss_incendio_intrus_1.png"
MODELFILENAME = "../Rede_cnn_testes_finais/modelo_final_intrusoes_incendio.pt"
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


def train_model(modelo):
    torch.manual_seed(42)
    modelo.to(device)  

    melhor_val_loss = float('inf') # Começa no infinito
    contador_paciencia = 0
    melhores_pesos = None          # Vai guardar a melhor versão da rede
    parar_treino = False           # Flag para quebrar os loops

    num_tot_imagens_intrus = 117000
    num_tot_imagens_incendio = 140000
    numero_de_epocas = 100                            

    pesos = torch.tensor([1.0, 1.2 ,1.5, 1.7, 1.3]).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=pesos)
    optimizer = optim.Adam(modelo.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=15, 
        threshold=MIN_DELTA
    )

    #Loaders
    train_dataset_intrus = CustomDataSet(csv_file=TRAIN_DATA_LABELS_INTRUS, num_imagens=num_tot_imagens_intrus, image_dir=TRAIN_DATA_DIR_INTRUS, transforms=transform, tipo_dataset="intrusos", offset=0)

    train_dataset_incendio = CustomDataSet(csv_file=TRAIN_DATA_LABELS_INCENDIO, num_imagens=num_tot_imagens_incendio, image_dir=TRAIN_DATA_DIR_INCENDIO, transforms=transform, tipo_dataset="incendios", offset=0)

    dataset_final = ConcatDataset([train_dataset_incendio, train_dataset_intrus])

    train_loader = DataLoader(
        dataset=dataset_final, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        generator=gerador_fixo
    )


    val_dataset_intrus = CustomDataSet(csv_file=VAL_DATA_LABEL_INTRUS, num_imagens=8000, image_dir=VAL_DATA_DIR_INTRUS, transforms=transform_teste, tipo_dataset="intrusos", offset=0)
    val_dataset_incendio = CustomDataSet(csv_file=TEST_DATA_LABEL_INCENDIO, num_imagens=8000, image_dir=TEST_DATA_DIR_INCENDIO, transforms=transform_teste, tipo_dataset="incendios", offset=0)


    val_dataset = ConcatDataset([val_dataset_intrus, val_dataset_incendio])

    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gerador_fixo
    )

    all_train_losses = []
    all_val_losses = []
    val_steps =[] 
    
    global_step = 0 

    for epoca in range(numero_de_epocas):

        if parar_treino:
            break 

        print(f"\n--- A iniciar Época {epoca + 1}/{numero_de_epocas} ---")


        modelo.train() 

        with tqdm(train_loader, desc=f"A treinar Época {epoca+1}", ncols=120) as pbar:
            for images, labels in pbar:
                global_step += 1
                
                images = images.to(device)
                labels = labels.to(device)

                outputs = modelo(images)  
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_train_losses.append(loss.item())
                
                #Validation
                if global_step % VAL_FREQ_BATCHES == 0:
                    modelo.eval() #Mete o modelo em modo de avaliação 
                    val_loss_acumulada = 0.0
                    
                    with torch.no_grad(): #Não precisamos de calcular gradientes para validação
                        for v_images, v_labels in val_loader:
                            v_images = v_images.to(device)
                            v_labels = v_labels.to(device)
                            
                            v_outputs = modelo(v_images)
                            v_loss = criterion(v_outputs, v_labels)
                            val_loss_acumulada += v_loss.item()
                    
                    #Média da loss no dataset de validação
                    avg_val_loss = val_loss_acumulada / len(val_loader)
                    all_val_losses.append(avg_val_loss)
                    val_steps.append(global_step)
                    
                    scheduler.step(avg_val_loss)
                    
                    #Atualiza a barra para mostrar ambas as Losses
                    pbar.set_postfix(T_Loss=f"{loss.item():.3f}", V_Loss=f"{avg_val_loss:.3f}")

                    if USAR_EARLY_STOPPING:
                        #Se a loss atual for menor que a melhor loss
                        if avg_val_loss < (melhor_val_loss - MIN_DELTA):
                            melhor_val_loss = avg_val_loss
                            contador_paciencia = 0
                            #Guardar uma cópia da rede neste exato momento perfeito
                            melhores_pesos = copy.deepcopy(modelo.state_dict())
                        else:
                            contador_paciencia += 1
                            if contador_paciencia >= PACIENCIA:
                                print(f"\n\nEarly Stopping, a validação não melhora há {PACIENCIA} passos.")
                                parar_treino = True
                                break #Sai do loop dos batches
                    
                    modelo.train() #Volta para modo de treino
                else:
                    #Se não for batch de validação, atualiza só a loss de treino na barra
                    pbar.set_postfix(T_Loss=f"{loss.item():.3f}")

    #Depois de Treinar, e no caso de early stopping, restaurar os pesos para a melhor versão encontrada
    if USAR_EARLY_STOPPING and melhores_pesos is not None:
        print(f"\n>> A restaurar os pesos do modelo para a melhor validação (Loss: {melhor_val_loss:.4f})")
        modelo.load_state_dict(melhores_pesos)

    plt.figure(figsize=(10, 5))
    plt.plot(all_train_losses, label="Train Loss", color='#1f77b4', alpha=0.7)
    plt.plot(val_steps, all_val_losses, label="Validation Loss", color='#ff7f0e', marker='o', linewidth=2)
    
    plt.xlabel("Iteração (Batches)")
    plt.ylabel("Loss")
    plt.title("Evolução da Loss: Treino vs Validação")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSSIMAGENAME) 
    print(f"\nGráfico da loss guardado como {LOSSIMAGENAME}")
    plt.close('all')

def save_model(model, string = 0):
    nome = string if isinstance(string, str) else "modelo.pt"
    torch.save(model.state_dict(),nome)

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

def avaliar_completa(modelo, loader, device, class_names, threshold, mostrar_erros=True, num_erros_para_mostrar=10):
    modelo.eval()
    
    all_preds = []         # lista para guardar todas as previsoes do modelo
    all_labels = []        # lista para guardar todas as labels reais
    erros_encontrados = [] # Lista para guardar imagens onde o modelo falhou

    correct_exact = 0
    total = 0

    correct_array = [0] * num_classes
    total_array = [0] * num_classes

    print(f"A iniciar avaliação Multi-Label (Threshold > {threshold})...")
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="A avaliar"):
            images = images.to(device)

            labels = labels.to(device)
            
            outputs = modelo(images)
            
            probs = torch.sigmoid(outputs)
            
            #Transforma probabilidades em 0 ou 1 consoante o threshold
            predicted = (probs > threshold).int()

            #Detetar o número de classes presentes nos dados atuais
            n_classes = labels.shape[1] 

            #Estatística de Accuracy Exata, só conta se acertar em todas as labels
            for i in range(len(labels)):
                if torch.equal(predicted[i], labels[i]):
                    correct_exact += 1

                elif mostrar_erros and len(erros_encontrados) < num_erros_para_mostrar:
                    img_erro = images[i].cpu()
                    lbl_real = labels[i].cpu()
                    lbl_pred = predicted[i].cpu()
                    erros_encontrados.append((img_erro, lbl_real, lbl_pred))
                
                if len(class_names) > 1:
                    
                    for j in range(n_classes):
                        if labels[i][j] == 1.0 :
                            total_array[j] += 1
                            if predicted[i][j] == 1.0:
                                correct_array[j] += 1

                
                total += 1
            
            #Guardar para cálculo de métricas
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    #Converter listas de arrays para um grande Array 2D
    np_preds = np.vstack(all_preds)
    np_labels = np.vstack(all_labels)
    
    acc_exata = 100 * correct_exact / total
    print(f'\n================ RESULTADOS ================')
    print(f'Accuracy: {acc_exata:.2f}% ({correct_exact}/{total} imagens perfeitamente corretas)')

    if len(class_names) > 1:
        
        for i in range(len(class_names)):
            if total_array[i] > 0:
                print(f'Classe {class_names[i]}: {100 * correct_array[i] / total_array[i]:.2f}% ({correct_array[i]}/{total_array[i]})')
            else:
                print(f'Classe {class_names[i]}: 0.00% (0/0)')

    print(f'============================================\n')
    
    print(f"--- Análise de Erros por Classe ---")
    
    precision_array, recall_array, f1_array = [], [],[]

    #Iterar por cada classe (0,1,2)
    for cls_idx in range(num_classes):
        nome_classe = class_names[cls_idx]
        
        #Extrair apenas a coluna referente a esta classe
        real_cls = np_labels[:, cls_idx]
        pred_cls = np_preds[:, cls_idx]
        
        #verifica se a imagem é desta classe
        era_esta_classe = (real_cls == 1)
        nao_era_esta_classe = (real_cls == 0)

        #verifica se a imagem foi advinhada como sendo desta classe
        previu_esta_classe = (pred_cls == 1)
        previu_outra_coisa = (pred_cls == 0)
        
        #Verdadeiros e Falsos
        tp_count = (era_esta_classe & previu_esta_classe).sum()
        fn_count = (era_esta_classe & previu_outra_coisa).sum()
        fp_count = (nao_era_esta_classe & previu_esta_classe).sum()
       
        
        total_real = era_esta_classe.sum()
        total_outros = nao_era_esta_classe.sum()
        
        fn_perc = (100 * fn_count / total_real) if total_real > 0 else 0
        fp_perc = (100 * fp_count / total_outros) if total_outros > 0 else 0
        
        #Precision, Recall, F1
        precision = 100 * tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = 100 * tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_array.append(precision)
        recall_array.append(recall)
        f1_array.append(f1)

        print(f"\n>> CLASSE {class_names[cls_idx]}: {nome_classe.upper()}")
        print(f"   - Falsos Negativos : {fn_count}/{total_real} ({fn_perc:.2f}%)")
        print(f"   - Falsos Positivos : {fp_count}/{total_outros} ({fp_perc:.2f}%)")
        print(f"   - Precisão : {precision:.2f}%")
        print(f"   - Recall : {recall:.2f}%")
        print(f"   - F1 Score : {f1:.2f}%")

    #Médias Globais
    if len(precision_array) > 1:
        print(f"\n>> VALORES MÉDIOS GLOBAIS (Macro):")
        print(f"   - Precisão Média : {sum(precision_array) / num_classes:.2f}%")
        print(f"   - Recall Médio : {sum(recall_array) / num_classes:.2f}%")
        print(f"   - F1 Score Médio : {sum(f1_array) / num_classes:.2f}%")

    if mostrar_erros and erros_encontrados:
        print(f"\nExemplos de erros (Real vs Previsto):")
        n = len(erros_encontrados)
        cols = 5
        rows = math.ceil(n / cols)
        
        plt.figure(figsize=(3 * cols, 3.5 * rows))
        
        for i, (img, real, pred) in enumerate(erros_encontrados):
            plt.subplot(rows, cols, i + 1)
            
            img_display = img.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1) # Evita cores estranhas
            plt.imshow(img_display)
            
            # Converter [1, 0, 1] em "People, Dog"s
            nomes_reais = [class_names[j] for j in range(num_classes) if real[j] == 1.0]
            nomes_preds = [class_names[j] for j in range(num_classes) if pred[j] == 1.0]
            
            txt_real = ", ".join(nomes_reais) if nomes_reais else "Nada"
            txt_pred = ", ".join(nomes_preds) if nomes_preds else "Nada"
            
            plt.title(f"Real: {txt_real}\nPrev: {txt_pred}", fontsize=9, color='red')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(ERRORSNAME)
        plt.close('all')

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

#modelo = RedeNeuronal(num_classes=num_classes).to(device)

modelo=load_model(MODELFILENAME)

print("A iniciar treino...")

#train_model(modelo)

end_time = time.time()
total_duration = end_time - start_time

print(f"\n\nTempo total de execução: {str(timedelta(seconds=int(total_duration)))}")

#save_model(modelo, MODELFILENAME)

test_dataset_intrus  = CustomDataSet(TEST_DATA_LABEL_INTRUS, 10000, TEST_DATA_DIR_INTRUS,tipo_dataset="intrusos", transforms=transform_teste, offset=0)
test_dataset_incendio = CustomDataSet(TEST_DATA_LABEL_INCENDIO, 10000, TEST_DATA_DIR_INCENDIO, tipo_dataset="incendios", transforms=transform_teste, offset=0)

test_dataset = ConcatDataset([test_dataset_intrus, test_dataset_incendio])

#Adicionado worker_init_fn e generator para previsibilidade do DataLoader de Teste
test_loader  = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    worker_init_fn=seed_worker,
    generator=gerador_fixo
)

print("--- A avaliar o modelo---")

teste_time_start = time.time()
avaliar_completa(modelo, test_loader, device, class_names, THRESHOLD)

teste_time_end = time.time()
teste_duration = teste_time_end - teste_time_start
print(f"\nTempo total de avaliação: {str(timedelta(seconds=int(teste_duration)))}, para {len(test_dataset)} imagens de teste.")