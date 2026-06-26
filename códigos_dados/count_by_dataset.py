import polars as pl

#Através dos csv finais, imprime um resumo do total de imagens por dataset e a contagem detalhada por label

def imprimir_resumo(csv_caminho):
    # le o csv com separador e força a coluna label a ser lida como string
    df = pl.read_csv(csv_caminho, separator=';', schema_overrides={"label": pl.String})
    
    # o nome do dataset de origem está sempre na primeira parte do FILENAME
    df = df.with_columns(
        pl.col("FILENAME").str.split("_").list.get(0).alias("Dataset_Origem")
    )
    
    total_por_dataset = df.group_by("Dataset_Origem").len().rename({"len": "Total_Imagens"})
    
    # passa para pandas porque o explode de caracteres é mais direto nesta parte
    df_pandas = df.select(["Dataset_Origem", "label"]).to_pandas()
    

    # separa cada caracter do label para uma linha própria, para contar classes individuais
    df_pandas['label'] = df_pandas['label'].apply(lambda x: list(str(x)))
    df_pandas = df_pandas.explode('label')
    
    # conta quantas vezes cada label aparece, por dataset de origem
    resumo_final = df_pandas.groupby(['Dataset_Origem', 'label']).size().reset_index(name='Contagem')
    
    print("\n" + "="*40)
    print("      RESUMO TOTAL POR DATASET")
    print("="*40)
    print(total_por_dataset.sort("Total_Imagens", descending=True))
    
    print("\n" + "="*40)
    print("   CONTAGEM DETALHADA POR LABEL")
    print("="*40)
    print(resumo_final.sort_values(['Dataset_Origem', 'label']))
    print("="*40)

imprimir_resumo("labels_final_train_shuffled.csv")
imprimir_resumo("labels_final_val.csv")
imprimir_resumo("labels_final_test.csv")