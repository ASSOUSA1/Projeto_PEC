#Remover datasets por prefixo de ficheiro

import os
import polars as pl
from tqdm import tqdm

def remover_dataset_por_prefixo(pasta_imagens, lista_de_csvs, prefixo_a_remover):

    print(f"A analisar os ficheiros para o prefixo '{prefixo_a_remover}'...")
    df_principal = pl.read_csv(lista_de_csvs[0], separator=';')
    
    #Filtra apenas as linhas onde o FILENAME começa com o prefixo
    ficheiros_a_apagar = df_principal.filter(
        pl.col("FILENAME").str.starts_with(prefixo_a_remover)
    )["FILENAME"].to_list()
    
    if len(ficheiros_a_apagar) == 0:
        print(f"Nenhum ficheiro encontrado com o prefixo '{prefixo_a_remover}'.")
        return

    print(f"Foram encontradas {len(ficheiros_a_apagar)} imagens para eliminar.")
    
    #Apagar as imagens físicas da pasta
    apagados_fisicamente = 0
    for nome_ficheiro in tqdm(ficheiros_a_apagar, desc="A apagar ficheiros físicos"):
        caminho_img = os.path.join(pasta_imagens, nome_ficheiro)
        if os.path.exists(caminho_img):
            os.remove(caminho_img)
            apagados_fisicamente += 1
            
    print(f"-> {apagados_fisicamente} imagens eliminadas do disco.")

    #Limpar as linhas em TODOS os CSVs (o normal e o shuffled)
    for csv_caminho in lista_de_csvs:
        if os.path.exists(csv_caminho):
            df_atual = pl.read_csv(csv_caminho, separator=';')
            
            #manter o que NÃO começa com o prefixo
            df_limpo = df_atual.filter(~pl.col("FILENAME").str.starts_with(prefixo_a_remover))
            
            df_limpo.write_csv(csv_caminho, separator=';')
            print(f"-> CSV atualizado: '{csv_caminho}' (restam {len(df_limpo)} linhas)")



prefixo_indesejado = "Objects365_train"  

pasta_com_imagens = "Dataset_intrus_final/train/images/images"

csvs_a_limpar =[
    "labels_final_test.csv"
    #"labels_final_train_shuffled.csv"
]

remover_dataset_por_prefixo(pasta_com_imagens, csvs_a_limpar, prefixo_indesejado)