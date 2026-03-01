# Databricks notebook source
!pip install -q tqdm

# COMMAND ----------

import requests
import os
from tqdm import tqdm

urls = [
    'https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/54e04a6b-15b3-4bda-a330-b8e805deabe4/download/dados_abertos_psr_2016a2024csv.csv',
    'https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/ac7e4351-974f-4958-9294-627c5cbf289a/download/dados_abertos_psr_2025csv.csv'
]

path = '/Volumes/00_raw/data/seguro_rural'

os.makedirs(path, exist_ok=True)

for url in tqdm(urls):
    filename = url.split('/')[-1]
    file_path = os.path.join(path, filename)
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

# COMMAND ----------

dbutils.fs.ls('/Volumes/00_raw/data/seguro_rural')
