## -*- coding: utf-8 -*-
import os
import os
import shutil
import warnings

#código para fazer download de datasets a partir do FiftyOne

venv_path = "/work/meu_env"

db_dir = "/work/fiftyone_db_coco"
zoo_dir = "/work/fiftyone_zoo_coco"

os.makedirs(db_dir, exist_ok=True)
os.makedirs(zoo_dir, exist_ok=True)

mongod_path = os.path.join(venv_path, "lib/python3.11/site-packages/fiftyone_db_bin/bin/mongod")

os.environ["FIFTYONE_DATABASE_DIR"] = db_dir
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = zoo_dir
os.environ["FIFTYONE_MONGOD_EXECUTABLE"] = mongod_path

import fiftyone as fo
import fiftyone.zoo as foz

warnings.filterwarnings("ignore")

pasta_exportacao_yolo = ".teste_dataset_intrus_coco/dataset"

#classes = ["Person", "Cat", "Dog"] # Open Images
classes = ["person", "cat", "dog"] # COCO

if os.path.exists(pasta_exportacao_yolo):
    shutil.rmtree(pasta_exportacao_yolo)

def limpar_memoria(nome):
    if fo.dataset_exists(nome):
        fo.delete_dataset(nome)


#Apagar o dataset da base de dados oculta do FiftyOne
nome_do_dataset = "coco"
if fo.dataset_exists(nome_do_dataset):
    fo.delete_dataset(nome_do_dataset)
    print("Dataset antigo apagado")


config_splits = {
    "train": 5000,
    "validation": 3000
}

for split, max_imagens in config_splits.items():

    todos_os_ids =[]

    #Para cada classe vai procurar X imagens que tenham pelo menos o label dessa classe
    #Vai guardar o nome da imagem na lista para depois fazer download da imagem com todos os labels

    for classe in classes:
        print(f"\n--- A escolher {max_imagens} imagens de {classe}... ---")
        
        nome_tmp = f"tmp_{classe}_{split}"
        
        # Saca x imagens APENAS desta classe
        ds_tmp = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            label_types=["detections"],
            classes=[classe],  
            max_samples=max_imagens,
            shuffle=True,
            dataset_name=nome_tmp
        )

        ids_tmp =[os.path.splitext(os.path.basename(p))[0] for p in ds_tmp.values("filepath")]
        todos_os_ids.extend(ids_tmp)

    #tranforma num set para eliminar repetidos e retorna para lista
    todos_os_ids = list(set(todos_os_ids))

    print("A carregar dataset")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=classes,            
        image_ids=todos_os_ids,     # Usa a lista criada com x iamgens de car classe
        dataset_name=f"dataset_final_{split}"
    )



    pasta_final = os.path.join("dataset_final_coco", split)
    pasta_final_abs = os.path.abspath(pasta_final)

    os.makedirs(pasta_final_abs, exist_ok=True)

    print(f"A exportar {split} para {pasta_final_abs}")

    dataset.export(
        export_dir=pasta_final_abs,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",           
        classes=classes ,
        export_media=True,
        overwrite=True
    )

    print(f"Exportação de {split} terminada.")

    # Limpar os datasets temporarios usados para escolher as imagens
    for classe in classes:
        limpar_memoria(f"tmp_{classe}_{split}")
    limpar_memoria(f"dataset_final_{split}")