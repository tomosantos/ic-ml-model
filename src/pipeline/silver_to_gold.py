# Databricks notebook source
# DBTITLE 1,Setup
import sys
from pyspark.sql import functions as F

sys.path.insert(0, '../lib')
from const import (
    TABLE_SILVER_CLEANED,
    TABLE_GOLD_FEATURES,
    TABLE_GOLD_LABELS,
    COLUNAS_FEATURES,
    COLUNAS_LABELS,
)

# COMMAND ----------

# DBTITLE 1,Leitura da Camada Silver
df = spark.read.table(TABLE_SILVER_CLEANED)

# COMMAND ----------

# DBTITLE 1,Garantir que o Catálogo/Schema Gold existe
# MAGIC %sql
# MAGIC CREATE CATALOG  IF NOT EXISTS 03_gold;
# MAGIC CREATE SCHEMA   IF NOT EXISTS 03_gold.seg_rural;

# COMMAND ----------

# DBTITLE 1,Separação Features / Labels
# ─────────────────────────────────────────────────────────────────────────────
# FEATURES: apenas preditores disponíveis NO MOMENTO DA CONTRATAÇÃO.
# Nenhuma variável de desfecho pós-contratual (sinistro, indenizacao, evento,
# sinistralidade) é incluída para evitar data leakage.
# ─────────────────────────────────────────────────────────────────────────────
colunas_features = [c for c in COLUNAS_FEATURES if c in df.columns]
df_features = df.select(colunas_features)

# ─────────────────────────────────────────────────────────────────────────────
# LABELS: variáveis resposta + chaves para join com a Feature Store.
# A referência temporal (dtRef) é dt_inicio_vigencia — ponto em que a apólice
# começa e as features devem ser calculadas.
# ─────────────────────────────────────────────────────────────────────────────
colunas_labels = [c for c in COLUNAS_LABELS if c in df.columns]
df_labels = df.select(colunas_labels)

# COMMAND ----------

# DBTITLE 1,Check
print(f"Features — colunas: {df_features.columns}")
print(f"Labels   — colunas: {df_labels.columns}")
print(f"Linhas: {df.count():,}")

display(df_features.limit(5))
display(df_labels.limit(5))

# COMMAND ----------

# DBTITLE 1,Escrita na Camada Gold
(
    df_features.write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(TABLE_GOLD_FEATURES)
)

(
    df_labels.write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(TABLE_GOLD_LABELS)
)

print(f"✓ Features gravadas em {TABLE_GOLD_FEATURES}  ({df_features.count():,} linhas)")
print(f"✓ Labels   gravadas em {TABLE_GOLD_LABELS}  ({df_labels.count():,} linhas)")

# COMMAND ----------
