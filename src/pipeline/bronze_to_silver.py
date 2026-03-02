# Databricks notebook source
# DBTITLE 1,Verificação das Colunas
df_historical = spark.read.table("01_bronze.seg_rural.historical_seg")
df_actual = spark.read.table("01_bronze.seg_rural.seg_2025")

df_historical.columns == df_actual.columns

# COMMAND ----------

# DBTITLE 1,Criação da Tabela Silver
# MAGIC %sql
# MAGIC -- 1. Configuração de Contexto
# MAGIC USE CATALOG 02_silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS 02_silver.seg_rural;
# MAGIC USE SCHEMA seg_rural;
# MAGIC
# MAGIC -- 2. Garantir que a tabela de destino existe
# MAGIC -- Criamos a estrutura baseada na união das tabelas da Bronze (vazia)
# MAGIC CREATE TABLE IF NOT EXISTS 02_silver.seg_rural.seg_cleaned
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT * FROM 01_bronze.seg_rural.historical_seg WHERE 1=0;
# MAGIC
# MAGIC -- 3. Preparação dos Dados (Origem)
# MAGIC CREATE OR REPLACE TEMP VIEW agg_seg AS 
# MAGIC WITH raw AS (
# MAGIC   SELECT * FROM 01_bronze.seg_rural.historical_seg
# MAGIC   UNION ALL
# MAGIC   SELECT * FROM 01_bronze.seg_rural.seg_2025
# MAGIC ),
# MAGIC dedup AS (
# MAGIC   -- Lógica para garantir que, se um ID aparecer em ambas, pegamos a versão mais recente
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY ID_PROPOSTA ORDER BY DT_PROPOSTA, DT_APOLICE DESC) AS rn
# MAGIC   FROM raw
# MAGIC )
# MAGIC SELECT
# MAGIC   * EXCEPT(rn)
# MAGIC FROM dedup
# MAGIC WHERE rn = 1;

# COMMAND ----------

# DBTITLE 1,Check
# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM agg_seg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tratamento

# COMMAND ----------

import unicodedata
from pyspark.sql.types import StringType
from pyspark.sql import functions as F

# Mapeamento de caracteres acentuados para sem acento
acentos = "áàâãéèêíìîóòôõúùûçÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ"
sem_acentos = "aaaaeeeiiioooouuucAAAAEEEIIIOOOOUUUC"

def remove_acentos_col(col_name):
  return F.translate(F.col(col_name), acentos, sem_acentos)

def normalize_str(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')

# COMMAND ----------

# DBTITLE 1,Cell 6
# 1. Captura a View existente
df = spark.table('agg_seg')

# 2. Substituir valores nas colunas
df = df.replace(['-', '...', 'X'], '0')

# 3. Limpa os NOMES das colunas (Removendo acentos e espaços nos cabeçalhos)
novos_nomes = [normalize_str(c) for c in df.columns]

for col in df.columns:
  df = df.withColumnRenamed(col, normalize_str(col))

# 3. Limpa os VALORES das colunas (Apenas colunas do tipo String)
from pyspark.sql.types import StringType

colunas_string = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

for col_name in colunas_string:
  df = df.withColumn(col_name, F.translate(F.col(col_name), acentos, sem_acentos))

display(df)


# COMMAND ----------


