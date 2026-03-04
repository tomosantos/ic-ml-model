# Databricks notebook source
# DBTITLE 1,Setup
# Este notebook é invocado pelo job seg_silver_to_gold para materializar cada
# tabela da Feature Store. Recebe o parâmetro `feature` via base_parameters.
#
# Uso manual:
#   dbutils.widgets.text('feature', 'fs_historico_municipio')
#   dbutils.widgets.text('dt_ref',  '')   # vazio = hoje
import sys
import datetime

from databricks.feature_engineering import FeatureEngineeringClient

sys.path.insert(0, '../lib')
from const import (
    TABLE_FS_HISTORICO_MUN,
    TABLE_FS_RISCO_CULTURA_UF,
    TABLE_FS_APOLICE_FINANCEIRO,
)

# COMMAND ----------

# DBTITLE 1,Parâmetros
dbutils.widgets.text('feature', 'fs_historico_municipio')
dbutils.widgets.text('dt_ref',  '')

feature = dbutils.widgets.get('feature')
dt_ref  = dbutils.widgets.get('dt_ref') or str(datetime.date.today())

print(f"feature : {feature}")
print(f"dt_ref  : {dt_ref}")

# COMMAND ----------

# DBTITLE 1,Mapeamento feature → tabela destino
FEATURE_MAP = {
    'fs_historico_municipio':  TABLE_FS_HISTORICO_MUN,
    'fs_risco_cultura_uf':     TABLE_FS_RISCO_CULTURA_UF,
    'fs_apolice_financeiro':   TABLE_FS_APOLICE_FINANCEIRO,
}

# Chaves primárias de cada tabela (necessárias para o FeatureEngineeringClient)
PRIMARY_KEYS = {
    'fs_historico_municipio':  ['dtRef', 'mun'],
    'fs_risco_cultura_uf':     ['dtRef', 'uf', 'tipo_cultura'],
    'fs_apolice_financeiro':   ['dtRef', 'apolice'],
}

if feature not in FEATURE_MAP:
    raise ValueError(f"Feature desconhecida: '{feature}'. Opções: {list(FEATURE_MAP.keys())}")

dest_table  = FEATURE_MAP[feature]
primary_key = PRIMARY_KEYS[feature]

# COMMAND ----------

# DBTITLE 1,Execução da Query e Materialização
sql_path = f'../{feature.replace("fs_", "feature_store/fs_")}.sql'
query    = open(sql_path).read().format(dt_ref=dt_ref)

df = spark.sql(query)

fe = FeatureEngineeringClient()

fe.create_table(
    name=dest_table,
    primary_keys=primary_key,
    df=df,
    description=f"Feature Store — {feature} | dt_ref={dt_ref}",
    schema=df.schema,
)

fe.write_table(
    name=dest_table,
    df=df,
    mode='merge',
)

print(f"✓ {dest_table} materializada com {df.count():,} linhas para dt_ref={dt_ref}")

# COMMAND ----------
