#Transformar dados divididos em pastas por classe em dados numa pasta com labels em CSV

import os
import shutil
import polars as pl
from tqdm import tqdm

def organizar_dataset_kaggle(pasta_origem, pasta_destino, prefixo):
    img_dest = os.path.join(pasta_destino, "images")
    os.makedirs(img_dest, exist_ok=True)
    
    # mapeia o nome da pasta (classe) para o label numérico usado no resto do projeto
    mapeamento = {
        "Cat": 1,
        "Dog": 2,
        "Person": 0
    }
    
    dados = []
    contador = 1
    
    print("A organizar e converter labels...")
    # percorre todas as pastas e subpastas a partir de pasta_origem
    for root, dirs, files in os.walk(pasta_origem):
        for file in tqdm(files):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                caminho_completo = os.path.join(root, file)
                
                nome_pasta = os.path.basename(root)
                
                label_final = mapeamento.get(nome_pasta, nome_pasta)
                
                # renomeia a imagem com um prefixo + número sequencial, para evitar repetidos quando se juntam várias origens de dados
                novo_nome = f"{prefixo}_{contador:05d}.jpg"
                
                shutil.copy(caminho_completo, os.path.join(img_dest, novo_nome))
                
                dados.append({"FILENAME": novo_nome, "label": label_final})
                
                contador += 1
                
    # Cria e guarda o CSV
    df = pl.DataFrame(dados)
    df.write_csv(os.path.join(pasta_destino, "dataset_intrus_dogvscat.csv"), separator=';')
    print(f"Sucesso! {contador-1} imagens organizadas em '{pasta_destino}'")


organizar_dataset_kaggle(
    pasta_origem="PetImages", 
    pasta_destino="PeTImages_formatado", 
    prefixo="dogvscat"
)