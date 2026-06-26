#Faz o mix do csv
import pandas as pd 

nome_csv = "labels_final_train"

print("A baralhar o CSV ...")
excel = pd.read_csv(nome_csv + ".csv", sep=';')
excel = excel.sample(frac=1).reset_index(drop=True)
excel.to_csv(nome_csv + "_shuffled.csv", sep=';', index=False)

print("CSV guardado em " + nome_csv + "_shuffled.csv")