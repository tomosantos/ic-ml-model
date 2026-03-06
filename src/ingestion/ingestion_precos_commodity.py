# Databricks notebook source
# DBTITLE 1,Setup
# Ingestão mensal de preços de commodities agrícolas — CEPEA/ESALQ
#
# Culturas monitoradas (>70% das apólices no SISSER):
#   soja, milho_1a_safra, milho_2a_safra, cafe
#
# Destino: 00_raw.mercado.precos_commodity_mensal
#   (ano INT, mes INT, cultura STRING, preco_rs_saca DOUBLE)
#
# Fonte alternativa automatizável: CONAB (via API pública) para validação cruzada.

!pip install -q requests pandas openpyxl

# COMMAND ----------
# DBTITLE 1,Imports
import sys
import io
import re
from datetime import datetime

import requests
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType

sys.path.insert(0, '../lib')
from const import (
    TABLE_RAW_PRECOS,
    CULTURAS_CEPEA,
    VOLUME_MERCADO,
)

# COMMAND ----------
# DBTITLE 1,Configuração dos endpoints CEPEA
# CEPEA disponibiliza séries históricas via download de planilhas.
# Endpoint de download mensal por indicador (GET com parâmetros de data).
# Produto ID: soja=6, milho=2, café arábica=30
# Unidade de referência: R$/saca 60 kg (soja/milho) | R$/saca 60 kg (café)

CEPEA_BASE = 'https://cepea.esalq.usp.br/br/consultas-ao-banco-de-dados-do-site.aspx'

# Mapeamento cultura → código de produto CEPEA e praça de referência
CEPEA_PRODUCTS = {
    'soja':         {'id': '6',  'desc': 'Soja Paraná (R$/sc 60kg)'},
    'milho_1a_safra': {'id': '2',  'desc': 'Milho Esalq/BM&F (R$/sc 60kg)'},
    'milho_2a_safra': {'id': '2',  'desc': 'Milho Esalq/BM&F (R$/sc 60kg)'},  # mesma série, safra identificada pelo mês
    'cafe':         {'id': '30', 'desc': 'Café Arábica (R$/sc 60kg)'},
}

# Datas da série histórica
DT_INICIO = '01/01/2016'
DT_FIM    = datetime.today().strftime('%d/%m/%Y')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; DataPipeline/1.0)',
    'Referer': CEPEA_BASE,
}

# COMMAND ----------
# DBTITLE 1,Função de download CEPEA
def download_cepea_serie(product_id: str, dt_ini: str, dt_fim: str) -> pd.DataFrame:
    """Baixa a série mensal de preços de um produto CEPEA.

    Tenta o endpoint de exportação XLSX da plataforma CEPEA.
    Retorna DataFrame com colunas ['data', 'preco_rs_saca'].
    """
    url = (
        f'https://cepea.esalq.usp.br/br/consultas-ao-banco-de-dados-do-site.aspx'
        f'?produto={product_id}&'
        f'data_inicio={dt_ini}&data_fim={dt_fim}&'
        f'formato=xlsx'
    )

    resp = requests.get(url, headers=HEADERS, timeout=60)

    if resp.status_code != 200 or len(resp.content) < 512:
        # Fallback: tenta o endpoint alternativo de CSV
        url_csv = url.replace('formato=xlsx', 'formato=csv')
        resp = requests.get(url_csv, headers=HEADERS, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Falha ao baixar CEPEA produto {product_id}: HTTP {resp.status_code}"
            )
        df = pd.read_csv(io.StringIO(resp.text), sep=';', decimal=',', skiprows=3)
    else:
        df = pd.read_excel(io.BytesIO(resp.content), skiprows=3)

    # Normaliza nomes de colunas independente do layout retornado
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Detecta coluna de data e coluna de preço à vista
    date_col  = next((c for c in df.columns if 'data' in c), None)
    price_col = next((c for c in df.columns if 'vista' in c or 'preco' in c or 'r$' in c), None)

    if date_col is None or price_col is None:
        raise ValueError(
            f"Colunas esperadas não encontradas no retorno CEPEA (produto {product_id}). "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df = df[[date_col, price_col]].rename(columns={date_col: 'data', price_col: 'preco_rs_saca'})
    df['data']          = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
    df['preco_rs_saca'] = (
        df['preco_rs_saca']
        .astype(str)
        .str.replace(r'[^\d,.]', '', regex=True)
        .str.replace(',', '.')
    )
    df['preco_rs_saca'] = pd.to_numeric(df['preco_rs_saca'], errors='coerce')
    df = df.dropna(subset=['data', 'preco_rs_saca'])
    return df

# COMMAND ----------
# DBTITLE 1,Download de todas as culturas e agregação mensal
all_frames = []

for cultura, cfg in CEPEA_PRODUCTS.items():
    print(f"Baixando {cultura} (id={cfg['id']}) …")
    try:
        df_raw = download_cepea_serie(cfg['id'], DT_INICIO, DT_FIM)

        # Agrega para granularidade mensal (média dos preços diários do mês)
        df_raw['ano'] = df_raw['data'].dt.year
        df_raw['mes'] = df_raw['data'].dt.month
        df_mensal = (
            df_raw
            .groupby(['ano', 'mes'], as_index=False)['preco_rs_saca']
            .mean()
        )
        df_mensal['cultura'] = cultura
        all_frames.append(df_mensal)
        print(f"  ✓ {len(df_mensal)} meses")

    except Exception as exc:
        print(f"  ✗ Falha para {cultura}: {exc}")

if not all_frames:
    raise RuntimeError("Nenhuma série foi baixada com sucesso. Verifique os endpoints CEPEA.")

df_precos = pd.concat(all_frames, ignore_index=True)
df_precos = df_precos[['ano', 'mes', 'cultura', 'preco_rs_saca']].sort_values(['cultura', 'ano', 'mes'])

print(f"\nTotal de registros: {len(df_precos):,}")
print(df_precos.groupby('cultura').size())

# COMMAND ----------
# DBTITLE 1,Escrita na tabela Delta
df_spark = spark.createDataFrame(df_precos) \
    .withColumn('ano',          F.col('ano').cast(IntegerType())) \
    .withColumn('mes',          F.col('mes').cast(IntegerType())) \
    .withColumn('cultura',      F.col('cultura').cast(StringType())) \
    .withColumn('preco_rs_saca', F.col('preco_rs_saca').cast(DoubleType()))

spark.sql("CREATE SCHEMA IF NOT EXISTS 01_bronze.seg_rural")

df_spark.write \
    .format('delta') \
    .mode('overwrite') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(TABLE_RAW_PRECOS)

print(f"✓ {TABLE_RAW_PRECOS} escrita com {df_spark.count():,} linhas")

# COMMAND ----------
# DBTITLE 1,Verificação
display(
    spark.table(TABLE_RAW_PRECOS)
        .orderBy('cultura', 'ano', 'mes')
)
