# Bibliotecas
# Pytorch e Torchvision
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

# Uteis
import math
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import os
import random  # Fixa a seed do Python built-in
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

# Função para garantir determinismo
def fixar_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Função para manter a consistência da seed nos processos paralelos (workers)
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
TRAIN_DATA_DIR = "../FireSmokeDS_upsampled_resized_aug/train/images"
TRAIN_DATA_LABELS = "../dataset_train_shuffled.csv"
TEST_DATA_DIR = "../FireSmokeDS_upsampled_resized_aug/val/images"
TEST_DATA_LABEL = "../dataset_val.csv"

os.makedirs("../Rede_cnn_testes_finais", exist_ok=True)

LOSSIMAGENAME = "../Rede_cnn_testes_finais/evolution_final_incendio.png"
MODELFILENAME = "../Rede_cnn_testes_finais/modelo_final_incendio.pt"
CONFUSIONMATRIXNAME = "../Rede_cnn_testes_finais/conf_final_incendio.png"
ERRORSNAME = "../Rede_cnn_testes_finais/errors_final_incendio.png"


BATCH_SIZE = 500

IMG_HEIGHT = 320
IMG_WIDTH = 320


VAL_FREQ_BATCHES = 50       
USAR_EARLY_STOPPING = True  
PACIENCIA = 20            
MIN_DELTA = 0.001           

class_names =[
    "Nada",           # Índice 0
    "Fumo ou Fogo"    # Índice 1
]

# Dicionário para converter o Índice (que sai do modelo) para Texto (para o print)
idx_to_label = {idx: name for idx, name in enumerate(class_names)}

# Número total de classes (será 2)
num_classes = len(class_names)


transform = transforms.v2.Compose([
    transforms.v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.v2.RandomHorizontalFlip(p=0.5), 
    transforms.v2.RandomRotation(degrees=10),
    transforms.v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

transform_teste = transforms.v2.Compose([
    transforms.v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

# Dataset personalizado
class CustomDataSet(Dataset):
    def __init__(self, csv_file, num_imagens, image_dir, transforms=None, offset=0):
        
        # separator é ; porque o csv foi guardado assim
        labels = pl.read_csv(csv_file, separator=';', columns=["FILENAME", "label"])
        if num_imagens is not None:
            labels = labels.slice(offset, num_imagens)

        self.labels = labels
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    # Devolve tuplo(imagem, label)
    def __getitem__(self, idx):
        file_name = self.labels[idx, "FILENAME"]
        
        label_original = int(self.labels[idx, "label"])
        label = 0 if label_original == 3 else 1
        
        complete_dir = os.path.join(self.image_dir, file_name)
        
        try:
            img = Image.open(complete_dir).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            return img, label
        except Exception as e:
            print(f"Erro ao carregar {file_name}: {e}")
            return torch.zeros((3, IMG_HEIGHT, IMG_WIDTH)), 0

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

        )

        self.avg_pool = nn.AdaptiveAvgPool2d(7)
        self.max_pool = nn.AdaptiveMaxPool2d(7)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7*2, 256),
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

    num_tot_imagens = 140000
    numero_de_epocas = 100

    pesos = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos)
    
    optimizer = optim.Adam(modelo.parameters(), lr=1e-4, weight_decay=1e-4 )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=15, 
        threshold=MIN_DELTA
    )

    print("\nA preparar Loaders (Lazy Loading)...")
    train_dataset = CustomDataSet(TRAIN_DATA_LABELS, num_imagens=num_tot_imagens, image_dir=TRAIN_DATA_DIR, transforms=transform, offset=0)
    
    # Configuração de workers e gerador para garantir determinismo nos batches de treino
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        generator=gerador_fixo
    )

    val_dataset = CustomDataSet(TEST_DATA_LABEL, num_imagens=5000, image_dir=TEST_DATA_DIR, transforms=transform_teste, offset=0)
    
    # Configuração de workers e gerador para garantir determinismo nos batches de validação
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gerador_fixo
    )

    melhor_val_loss = float('inf')
    contador_paciencia = 0
    melhores_pesos = None
    parar_treino = False

    all_train_losses = []
    all_val_losses = []
    val_steps =[]
    global_step = 0 

    for epoca in range(numero_de_epocas):
        if parar_treino:
            break

        print(f"\n--- A iniciar Época {epoca + 1}/{numero_de_epocas} ---")


        modelo.train()

        with tqdm(train_loader, desc=f"Step Época {epoca+1}", ncols=120) as pbar:
            for images, labels in pbar:
                global_step += 1
                
                images = images.to(device)
                labels = labels.to(device) # Já vem binário do Dataset

                outputs = modelo(images)  
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_train_losses.append(loss.item())

                if global_step % VAL_FREQ_BATCHES == 0:
                    modelo.eval()
                    val_loss_acumulada = 0.0
                    
                    with torch.no_grad():
                        for v_images, v_labels in val_loader:
                            v_images = v_images.to(device)
                            v_labels = v_labels.to(device)
                            v_outputs = modelo(v_images)
                            v_loss = criterion(v_outputs, v_labels)
                            val_loss_acumulada += v_loss.item()
                    
                    avg_val_loss = val_loss_acumulada / len(val_loader)
                    all_val_losses.append(avg_val_loss)
                    val_steps.append(global_step)

                    scheduler.step(avg_val_loss)
                    
                    lr_atual = optimizer.param_groups[0]['lr']
                    
                    pbar.set_postfix(T_Loss=f"{loss.item():.4f}", V_Loss=f"{avg_val_loss:.4f}")

                    if USAR_EARLY_STOPPING:
                        if avg_val_loss < (melhor_val_loss - MIN_DELTA):
                            melhor_val_loss = avg_val_loss
                            contador_paciencia = 0
                            melhores_pesos = copy.deepcopy(modelo.state_dict())
                        else:
                            contador_paciencia += 1
                            if contador_paciencia >= PACIENCIA:
                                print(f"\n[ALERTA] Early Stopping ativado! Validação não melhora há {PACIENCIA} passos.")
                                parar_treino = True
                                break 

                    modelo.train()
                else:
                    pbar.set_postfix(T_Loss=f"{loss.item():.4f}")

    if USAR_EARLY_STOPPING and melhores_pesos is not None:
        print(f"\n>> A restaurar os pesos do modelo para a melhor validação (Loss: {melhor_val_loss:.4f})")
        modelo.load_state_dict(melhores_pesos)

    plt.figure(figsize=(10, 5))
    plt.plot(all_train_losses, label="Train Loss", color='#1f77b4', alpha=0.7)
    plt.plot(val_steps, all_val_losses, label="Validation Loss", color='#ff7f0e', marker='o', linewidth=2)
    plt.xlabel("Iteração")
    plt.ylabel("Loss")
    plt.title("Evolução da Loss durante o treino")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSSIMAGENAME) 
    print("Gráfico da loss guardado como ", LOSSIMAGENAME)
    plt.close('all')

def save_model(model, string = 0):
    nome = string if isinstance(string, str) else "modelo.pt"
    torch.save(model.state_dict(),nome)

def load_model(modelo=0):

    model = RedeNeuronal(num_classes).to(device)

    caminho = modelo
    try:
        model.load_state_dict(torch.load(caminho, weights_only=True))
        model.to(device)
    except Exception as e:
        print(f"{e}") 
        exit()

    return model

def avaliar_completa(modelo, loader, device, class_names, mostrar_matriz=True, mostrar_erros=True, num_erros_para_mostrar=10):
    modelo.eval()  # Modo de avaliação (desliga Dropout, BatchNorm, etc)
    
    all_preds = []         # lista para guardar todas as previsões do modelo
    all_labels =[]        # lista para guardar todas as labels reais
    erros_encontrados =[] # Lista para guardar imagens onde o modelo falhou
    
    correct = 0
    total = 0
    
    print("A iniciar avaliação completa...")
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="A avaliar"):
            images = images.to(device)
            
            # Converter labels de string para inteiros
            # Se vier "0", "1", "2" transforma em tensor[0, 1, 2]
            labels_int = torch.tensor([int(l) for l in labels], dtype=torch.long).to(device)
            
            outputs = modelo(images)
            
            # Obter a classe com maior probabilidade (argmax)
            _, predicted = torch.max(outputs.data, 1)
            
            # Estatísticas
            total += labels_int.size(0)
            correct += (predicted == labels_int).sum().item()
            
            # Guardar para a matriz de confusão
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_int.cpu().numpy())
            
            # Guardar exemplos de erros
            if mostrar_erros and len(erros_encontrados) < num_erros_para_mostrar:
                # Índices dentro do batch onde a previsão foi diferente da realidade
                indices_erros = (predicted != labels_int).nonzero()
                
                for idx in indices_erros:
                    if len(erros_encontrados) >= num_erros_para_mostrar:
                        break
                    
                    idx = idx.item()
                    img_erro = images[idx].cpu()
                    lbl_real = labels_int[idx].item()
                    lbl_pred = predicted[idx].item()
                    
                    erros_encontrados.append((img_erro, lbl_real, lbl_pred))


    acc = 100 * correct / total
    print(f'\n================ RESULTADOS ================')
    print(f'Accuracy Final: {acc:.2f}% ({correct}/{total} imagens corretas)')
    print(f'============================================\n')
    

    # Converter listas para Arrays do Numpy 
    np_labels = np.array(all_labels) # Lista do que era real
    np_preds = np.array(all_preds)   # Lista do que o modelo previu
    
    print(f"--- Análise de Erros por Classe ---")
    
    # Classes a analisar
    classes_interesse = [1] 

    precision_array = []
    recall_array = []
    f1_array =[]

    for cls_idx in classes_interesse:
        nome_classe = class_names[cls_idx]
        
        # Verifica se a imagem é desta classe
        era_esta_classe = (np_labels == cls_idx)
        nao_era_esta_classe = (np_labels != cls_idx)
        
        # Verifica se a imagem foi advinhada como sendo desta classe
        previu_esta_classe = (np_preds == cls_idx)
        previu_outra_coisa = (np_preds != cls_idx)
         
        # Calcula Falsos Negativos
        #se era esta classe e previu outra coisa = falso negativo
        fn_count = (era_esta_classe & previu_outra_coisa).sum()
        
        # Quantas imagens desta classe existiam no total?
        total_real = era_esta_classe.sum()
        
        if total_real > 0:
            fn_perc = 100 * fn_count / total_real
        else:
            fn_perc = 0.0

        # Calcular Falsos Positivos
        #se não era esta classe e previu esta classe = falso positivo
        fp_count = (nao_era_esta_classe & previu_esta_classe).sum()
        
        # Quantas imagens existiam que NÃO eram desta classe?
        total_outros = nao_era_esta_classe.sum() 
        
        if total_outros > 0:
            fp_perc = 100 * fp_count / total_outros
        else:
            fp_perc = 0.0

        # Calcular precision, recall e f1-score

        tp_count = (era_esta_classe & previu_esta_classe).sum() # Verdadeiros Positivos

        precision = 100* tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0

        recall = 100* tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

        f1 = 2*precision*recall/(precision+recall) if (precision + recall) > 0 else 0

        precision_array.append(precision)
        recall_array.append(recall)
        f1_array.append(f1)


        print(f"\n>> CLASSE {cls_idx}: {nome_classe.upper()}")
        print(f"   - Falsos Negativos : {fn_count}/{total_real} ({fn_perc:.2f}%)")
        print(f"   - Falsos Positivos :  {fp_count}/{total_outros} ({fp_perc:.2f}%)")
        #print("\n")
        print(f"   - Precisão : {precision:.2f}%")
        print(f"   - Recall : {recall:.2f}%")
        print(f"   - F1 : {f1:.2f}%")

    print("\n")

    if len(precision_array) > 0:
        precision = sum(precision_array) / len(precision_array)
        recall = sum(recall_array) / len(recall_array)
        f1 = sum(f1_array) / len(f1_array)
    
    else:
        precision = recall = f1 = 0

    print(f"\n>> Valores médios :")
    print(f"   - Precisão : {precision:.2f}%")
    print(f"   - Recall : {recall:.2f}%")
    print(f"   - F1 : {f1:.2f}%")



    if mostrar_matriz:
        cm = confusion_matrix(all_labels, all_preds)
        cm_log = np.log1p(cm)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_log, annot=cm, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('O que o modelo previu')
        plt.ylabel('O que era na realidade')
        plt.title('Matriz de Confusão')
        plt.savefig(CONFUSIONMATRIXNAME)
        plt.close('all')

    if mostrar_erros and erros_encontrados:
        print(f"\nExemplos de erros (Real vs Previsto):")
        
        # Calcular linhas e colunas para o plot
        n = len(erros_encontrados)
        cols = 5
        rows = math.ceil(n / cols)
        
        plt.figure(figsize=(3 * cols, 3.5 * rows))
        
        for i, (img, real, pred) in enumerate(erros_encontrados):
            plt.subplot(rows, cols, i + 1)
            
            # Permute necessário: (C, H, W) -> (H, W, C) para o matplotlib
            img_display = img.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            plt.imshow(img_display)
            
            nome_real = class_names[real]
            nome_pred = class_names[pred]
            
            plt.title(f"Real: {nome_real}\nPrev: {nome_pred}", fontsize=9, color='red')
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
                    real_encontrada = True
            else:
                pass
                
        except Exception as e:
            pass
    
    # Apresentar Resultados na Consola de forma dinâmica
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
    
    # Construir título da imagem dinamicamente
    if real_encontrada:
        titulo = f"Real: {label_real_str}\nPrevisto: {label_prevista_str}"
    else:
        titulo = f"Previsto: {label_prevista_str}"
        
    plt.title(titulo, fontsize=12, fontweight='bold', color='blue')
    plt.axis('off')
    plt.savefig(ERRORSNAME)
    plt.close('all')


#modelo = RedeNeuronal(num_classes=num_classes).to(device)

#modelo=load_model(MODELFILENAME)
modelo=load_model("código_deuc/codigos_finais/modelo_final_incendio.pt")

print("A iniciar treino...")

#train_model(modelo)

#end_time = time.time()
#total_duration = end_time - start_time

#print(f"\n\nTempo total de execução: {str(timedelta(seconds=int(total_duration)))}")

#save_model(modelo, MODELFILENAME)

#test_dataset  = CustomDataSet(TEST_DATA_LABEL, 10000, TEST_DATA_DIR, transform_teste, 0)


#test_loader  = DataLoader(
#dataset=test_dataset, 
#batch_size=BATCH_SIZE, 
#shuffle=False, 
#num_workers=4, 
#pin_memory=True, 
#prefetch_factor=2,
#worker_init_fn=seed_worker,
#generator=gerador_fixo
#)

print()
print("--- A avaliar o modelo---")

#teste_time_start = time.time()


#avaliar_completa(modelo, test_loader, device, class_names)


#teste_time_end = time.time()
#teste_duration = teste_time_end - teste_time_start
#print(f"\nTempo total de avaliação: {str(timedelta(seconds=int(teste_duration)))}, para {len(test_dataset)} imagens de teste.")

#avaliar_imagem(modelo, "FireSmokeDS_upsampled_resized_aug/val/images/D-Fire_17221.jpg", device, class_names, transforms=transform_teste)

avaliar_imagem(modelo, "código_deuc/codigos_finais/image.png", device, class_names, transform_teste)