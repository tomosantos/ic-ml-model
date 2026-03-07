# Databricks notebook source
# MAGIC %pip install -q databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup
# Este notebook é invocado pelo job seg_silver_to_gold para materializar cada
# tabela da Feature Store. Recebe o parâmetro `feature` via base_parameters.
#
# Uso manual:
#   dbutils.widgets.text('feature', 'fs_historico_municipio')
import sys

from databricks.feature_engineering import FeatureEngineeringClient

sys.path.insert(0, '../lib')
from const import (
    TABLE_FS_HISTORICO_MUN,
    TABLE_FS_RISCO_CULTURA_UF,
    TABLE_FS_APOLICE_FINANCEIRO,
    TABLE_FS_RISCO_SEGURADORA_CULTURA,
    TABLE_FS_ANOMALIA_TAXA,
    TABLE_FS_CONCENTRACAO_CARTEIRA,
)

# COMMAND ----------

# DBTITLE 1,Parâmetros
dbutils.widgets.text('feature', 'fs_historico_municipio')

feature = dbutils.widgets.get('feature')

print(f"feature: {feature}")

# COMMAND ----------

# DBTITLE 1,Mapeamento feature → tabela destino
FEATURE_MAP = {
    'fs_historico_municipio':       TABLE_FS_HISTORICO_MUN,
    'fs_risco_cultura_uf':          TABLE_FS_RISCO_CULTURA_UF,
    'fs_apolice_financeiro':        TABLE_FS_APOLICE_FINANCEIRO,
    'fs_risco_seguradora_cultura':  TABLE_FS_RISCO_SEGURADORA_CULTURA,
    'fs_anomalia_taxa':             TABLE_FS_ANOMALIA_TAXA,
    'fs_concentracao_carteira':     TABLE_FS_CONCENTRACAO_CARTEIRA,
}

# Chaves primárias de cada tabela (necessárias para o FeatureEngineeringClient)
PRIMARY_KEYS = {
    'fs_historico_municipio':       ['dtRef', 'mun'],
    'fs_risco_cultura_uf':          ['dtRef', 'uf', 'tipo_cultura'],
    'fs_apolice_financeiro':        ['dtRef', 'apolice'],
    'fs_risco_seguradora_cultura':  ['dtRef', 'seguradora', 'tipo_cultura'],
    'fs_anomalia_taxa':             ['dtRef', 'cultura', 'uf'],
    'fs_concentracao_carteira':     ['dtRef', 'seguradora', 'mun'],
}

if feature not in FEATURE_MAP:
    raise ValueError(f"Feature desconhecida: '{feature}'. Opções: {list(FEATURE_MAP.keys())}")

dest_table  = FEATURE_MAP[feature]
primary_key = PRIMARY_KEYS[feature]

# COMMAND ----------

# DBTITLE 1,Execução da Query e Materialização
sql_path = f'../{feature.replace("fs_", "feature_store/fs_")}.sql'
query    = open(sql_path).read()

df = spark.sql(query)

fe = FeatureEngineeringClient()

fe.create_table(
    name=dest_table,
    primary_keys=primary_key,
    df=df,
    description=f"Feature Store — {feature}",
    schema=df.schema,
)

fe.write_table(
    name=dest_table,
    df=df,
    mode='merge',
)

print(f"✓ {dest_table} materializada com {df.count():,} linhas")
