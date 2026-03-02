# Databricks notebook source
!pip install -q tqdm

# COMMAND ----------

import requests
import os
from tqdm import tqdm

urls = [
    'https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/e6f95018-6c19-426a-9a62-fc9e5bfc721b/download/dados_abertos_psr_2016a2024.xlsx',
    'https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/b904117d-b758-406d-92ef-4c7762017c61/download/dados_abertos_psr_2025.xlsx'
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
