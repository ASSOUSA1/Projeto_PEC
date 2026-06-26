#Adicionar imagens a uma mesma pasta de imagens e atualizar o CSV com os labels correspondentes

import os
import shutil
import polars as pl
from tqdm import tqdm

def adicionar_ao_dataset(pasta_origem, pasta_destino, csv_origem, prefixo):
    # Apenas pasta de imagens
    img_dest = os.path.join(pasta_destino, "images")
    csv_final = os.path.join("labels_final_train.csv")
    
    os.makedirs(img_dest, exist_ok=True)
    
    offset = 0
    
    # Ler CSV do lote novo
    df_novo_lote = pl.read_csv(csv_origem, separator=';')
    novos_dados = []
    
    ficheiros = sorted([f for f in os.listdir(pasta_origem) if f.lower().endswith(('.jpg'))])
    
    print(f"-> A adicionar {len(ficheiros)} imagens com prefixo '{prefixo}'...")
    
    for i, nome_antigo in enumerate(tqdm(ficheiros, ncols=100)):
        num = offset + i + 1
        novo_nome = f"{prefixo}_{num:05d}"
        
        ext = os.path.splitext(nome_antigo)[1]
        nome_imagem_novo = f"{novo_nome}{ext}"
        
        #Copiar imagem
        shutil.copy(os.path.join(pasta_origem, nome_antigo), os.path.join(img_dest, nome_imagem_novo))
        
        #Atualizar CSV
        label_atual = df_novo_lote.filter(pl.col("FILENAME") == nome_antigo)["label"].item()
        novos_dados.append({"FILENAME": nome_imagem_novo, "label": label_atual})
    
    # Adicionar ao CSV final
    df_para_append = pl.DataFrame(novos_dados)
    
    if os.path.exists(csv_final):
        df_existente = pl.read_csv(csv_final, separator=';')
        df_final = pl.concat([df_existente, df_para_append])
    else:
        df_final = df_para_append
        
    df_final.write_csv(csv_final, separator=';')
    print(f"Sucesso! Dataset atualizado. Total de imagens no CSV: {len(df_final)}")

# adicionar_ao_dataset(pasta_origem, pasta_destino, csv_origem, prefixo)

#adicionar_ao_dataset("Datasets_pre_mix/Datasets_finais/dataset_intrus_openimagensV7/validation/images/val", "Dataset_intrus_final/validation/images", "dataset_val_intrusoes_openimages.csv", "OPENIMAGES_validation")

#adicionar_ao_dataset("Datasets_pre_mix/Datasets_finais/dataset_intrus_coco/validation/images/val", "Dataset_intrus_final/validation/images", "dataset_val_intrusoes_coco.csv", "COCO_validation")

#adicionar_ao_dataset("Datasets_pre_mix/Datasets_finais/PetImages_formatado/images", "Dataset_intrus_final/train/images", "dataset_intrus_dogvscat.csv", "dogvscat")

#adicionar_ao_dataset("Datasets_pre_mix/dataset_final_coco/test/images/val", "Dataset_intrus_final/test/images", "dataset_teste_intrusoes_OpenImages.csv", "OpenImages_test")

adicionar_ao_dataset("Datasets_pre_mix/Datasets_finais/Objects365_filtrado/images", "Dataset_intrus_final/train/images", "dataset_train_objects365.csv", "Objects365_train")