# Databricks notebook source
# DBTITLE 1,Setup
# Ingestão da série histórica do ONI (Oceanic Niño Index) — NOAA/CPC
#
# Fonte: https://cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
# Destino: 01_bronze.seg_rural.oni_mensal  (ano INT, mes INT, oni_valor DOUBLE, fase STRING)
#
# Fase:
#   el_nino  →  ONI >= +0.5
#   la_nina  →  ONI <= -0.5
#   neutro   →  -0.5 < ONI < +0.5

!pip install -q requests beautifulsoup4 lxml

# COMMAND ----------
# DBTITLE 1,Imports
import sys
import os

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType

sys.path.insert(0, '../lib')
from const import (
    ONI_URL,
    ONI_THRESHOLD,
    TABLE_RAW_ONI,
    VOLUME_CLIMA,
)

# COMMAND ----------
# DBTITLE 1,Download e parse do HTML
# Cada linha da tabela tem: Year | DJF | JFM | FMA | MAM | AMJ | MJJ | JJA | JAS | ASO | SON | OND | NDJ
# A coluna sazonal de 3 letras indica as iniciais dos 3 meses envolvidos;
# o mês central é usado como referência (ex.: DJF → janeiro = mês 1).

SEASON_TO_MONTH = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4,
    'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
    'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12,
}

print(f"Baixando ONI de {ONI_URL} …")
response = requests.get(ONI_URL, timeout=30)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'lxml')

# A tabela principal tem cabeçalho com 'Year' na primeira célula
oni_table = None
for table in soup.find_all('table'):
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    if 'Year' in headers and 'DJF' in headers:
        oni_table = table
        break

if oni_table is None:
    raise ValueError("Tabela ONI não encontrada na página NOAA. Verifique o layout de ONI_URL.")

headers = [th.get_text(strip=True) for th in oni_table.find_all('th')]
print(f"Colunas encontradas: {headers}")

rows = []
for tr in oni_table.find_all('tr')[1:]:   # pula cabeçalho
    cells = [td.get_text(strip=True) for td in tr.find_all('td')]
    if len(cells) < 2:
        continue
    rows.append(cells)

df_oni_wide = pd.DataFrame(rows, columns=headers[:len(rows[0])])

# COMMAND ----------
# DBTITLE 1,Melt para formato longo (ano, mes, oni_valor, fase)
df_oni_wide['Year'] = pd.to_numeric(df_oni_wide['Year'], errors='coerce')
df_oni_wide = df_oni_wide.dropna(subset=['Year'])
df_oni_wide['Year'] = df_oni_wide['Year'].astype(int)

season_cols = [c for c in df_oni_wide.columns if c in SEASON_TO_MONTH]

df_oni_long = df_oni_wide.melt(
    id_vars=['Year'],
    value_vars=season_cols,
    var_name='season',
    value_name='oni_valor_str',
)

df_oni_long['mes']       = df_oni_long['season'].map(SEASON_TO_MONTH)
df_oni_long['ano']       = df_oni_long['Year']
df_oni_long['oni_valor'] = pd.to_numeric(df_oni_long['oni_valor_str'], errors='coerce')

# Remove linhas ainda sem valor (ano ainda em curso sem média completa)
df_oni_long = df_oni_long.dropna(subset=['oni_valor'])

def classify_fase(v: float) -> str:
    if v >= ONI_THRESHOLD:
        return 'el_nino'
    if v <= -ONI_THRESHOLD:
        return 'la_nina'
    return 'neutro'

df_oni_long['fase'] = df_oni_long['oni_valor'].apply(classify_fase)

df_oni_final = df_oni_long[['ano', 'mes', 'oni_valor', 'fase']].sort_values(['ano', 'mes'])

print(f"Registros ONI: {len(df_oni_final):,} ({df_oni_final['ano'].min()}–{df_oni_final['ano'].max()})")

# COMMAND ----------
# DBTITLE 1,Conversão para Spark e escrita na tabela Delta
df_spark = spark.createDataFrame(df_oni_final) \
    .withColumn('ano',       F.col('ano').cast(IntegerType())) \
    .withColumn('mes',       F.col('mes').cast(IntegerType())) \
    .withColumn('oni_valor', F.col('oni_valor').cast(DoubleType())) \
    .withColumn('fase',      F.col('fase').cast(StringType()))

# Garante que o schema do catálogo é criado (Unity Catalog)
spark.sql("CREATE SCHEMA IF NOT EXISTS 01_bronze.seg_rural")

df_spark.write \
    .format('delta') \
    .mode('overwrite') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(TABLE_RAW_ONI)

print(f"✓ {TABLE_RAW_ONI} escrita com {df_spark.count():,} linhas")

# COMMAND ----------
# DBTITLE 1,Verificação
display(spark.table(TABLE_RAW_ONI).orderBy('ano', 'mes'))
