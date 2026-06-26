#Usado para formato yolo, para extrair classes de ficherios txt e gerar um csv com o nome do ficheiro e a label final fumo e fogo

import os
import csv
from tqdm import tqdm

def gerar_tabela_labels(pasta_imagens, pasta_labels, nome_ficheiro_saida):
    
    dados_para_salvar = []

    #verifica se a pasta existe
    
    if not os.path.exists(pasta_imagens):
        print(f"Erro: A pasta '{pasta_imagens}' não existe.")
        return

    ficheiros = os.listdir(pasta_imagens)
    print(f"A processar {len(ficheiros)} ficheiros...")

    #Loop para analisar cada ficheiro de imagem e a sua label

    for nome_imagem in tqdm(ficheiros, desc="A processar"):
        
        nome_base = os.path.splitext(nome_imagem)[0]
        caminho_label = os.path.join(pasta_labels, nome_base + ".txt")
        
        labels_encontrados = set() 
        
    #Ler o ficheiro de labels se existir
    #Vê todas as linhas e retira o primeiro argumento (0 ou 1)
        if os.path.exists(caminho_label):
            try:
                with open(caminho_label, 'r', encoding='utf-8') as f:
                    for linha in f:
                        linha = linha.strip()
                        if linha:
                            partes = linha.split()
                            if len(partes) > 0:
                                # Adiciona o primeiro caracter (0 ou 1) ao conjunto
                                labels_encontrados.add(partes[0])
                            
            except Exception as erro:
                tqdm.write(f"Erro ao ler {caminho_label}: {erro}")

    #Classificação da imagem com base nos labels encontrados
    #Verifica o que está dentro do set e atribui o label final
        
        tem_zero = "0" in labels_encontrados
        tem_um = "1" in labels_encontrados
        
        label_final = ""

        if tem_zero and tem_um:
            label_final = "2"  # Tem 0 e 1
        elif tem_zero:
            label_final = "0"  # Apenas 0
        elif tem_um:
            label_final = "1"  # Apenas 1
        else:
            label_final = "3"  # Vazio 

    #Guardar na lista (Indentação corrigida: agora executa para todas as imagens)
        dados_para_salvar.append([nome_imagem, label_final])

    #Guardar no ficheiro CSV
    try:
        with open(nome_ficheiro_saida, 'w', newline='', encoding='utf-8') as csvfile:
            # Usa ponto e vírgula como separador
            escritor = csv.writer(csvfile, delimiter=';')
            
            escritor.writerow(['FILENAME', 'label'])
            escritor.writerows(dados_para_salvar)
            
        print(f"Ficheiro '{nome_ficheiro_saida}' criado.")
        
    except Exception as e:
        print(f"Erro ao guardar CSV: {e}")

if __name__ == "__main__":
    PASTA_IMG = "FireSmokeDS_upsampled_resized_aug/train/images"  
    PASTA_LBL = "FireSmokeDS_upsampled_resized_aug/train/labels"   
    SAIDA = "dataset_train.csv"
    
    gerar_tabela_labels(PASTA_IMG, PASTA_LBL, SAIDA)