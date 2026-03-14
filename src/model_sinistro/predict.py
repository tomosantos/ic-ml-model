# Databricks notebook source
# MAGIC %pip install -q databricks-feature-engineering mlflow
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup
import re
import sys

import mlflow
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow.tracking import MlflowClient

sys.path.insert(0, '../lib')
from const import (
    TABLE_FS_ANOMALIA_TAXA,
    TABLE_FS_APOLICE_FINANCEIRO,
    TABLE_FS_CONCENTRACAO_CARTEIRA,
    TABLE_FS_HISTORICO_MUN,
    TABLE_FS_RISCO_CULTURA_UF,
    TABLE_FS_RISCO_SEGURADORA_CULTURA,
    TABLE_PREDICOES
)

sys.path.insert(0, '../model_sinistro')
from preprocessing import derive_features

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text('date', '')
dbutils.widgets.text('model_version', '7')

CUTOFF_OOT = pd.Timestamp('2025-01-01')

date          = dbutils.widgets.get('date')
model_version = dbutils.widgets.get('model_version')

if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
    raise ValueError(f"Formato de data inválido: '{date}'. Use YYYY-MM-DD.")

if pd.Timestamp(date) < CUTOFF_OOT:
    raise ValueError(
        f"predict.py é reservado para scoring OOT (>= {CUTOFF_OOT.date()}). "
        f"Recebido: '{date}'. Para avaliar datas anteriores, use o split OOS do train.py."
    )

if model_version != 'latest' and not re.match(r'^\d+$', model_version):
    raise ValueError(f"model_version deve ser 'latest' ou um inteiro positivo. Recebido: '{model_version}'")

model_name = '04_feature_store.seg_rural.sinistro'

# COMMAND ----------

# DBTITLE 1,Carregar Modelo
model_uri    = f'models:/{model_name}/{model_version}'
model_pyfunc = mlflow.pyfunc.load_model(model_uri)
run_id       = model_pyfunc.metadata.run_id
model        = mlflow.sklearn.load_model(f'runs:/{run_id}/model')

if model_version == 'latest':
    versions       = MlflowClient().get_latest_versions(model_name)
    actual_version = int(versions[0].version)
else:
    actual_version = int(model_version)

print(f'✓ Modelo carregado: {model_name} v{actual_version}  (run_id={run_id})')

# COMMAND ----------

# DBTITLE 1,Âncora de Scoring
# Apólices vigentes no período de interesse:
#   dtRef  = primeiro dia do mês informado
#   vigência ainda aberta na data de referência
df_anchor = spark.sql(f"""
    SELECT
        apolice,
        DATE_TRUNC('MONTH', dt_inicio_vigencia) AS dtRef,
        mun,
        uf,
        cultura,
        tipo_cultura,
        seguradora,
        regiao
    FROM 02_silver.seg_rural.seg_cleaned
    WHERE DATE_TRUNC('MONTH', dt_inicio_vigencia) = DATE('{date}')
      AND dt_fim_vigencia > '{date}'
""")

count_anchor = df_anchor.count()
assert count_anchor > 0, f"Nenhuma apólice vigente encontrada para date='{date}'"
print(f'✓ Âncora: {count_anchor:,} apólices')

# COMMAND ----------

# DBTITLE 1,Resolver FeatureLookups
fe = FeatureEngineeringClient()

feature_lookups = [
    FeatureLookup(
        table_name=TABLE_FS_HISTORICO_MUN,
        lookup_key=['dtRef', 'mun'],
    ),
    FeatureLookup(
        table_name=TABLE_FS_RISCO_CULTURA_UF,
        lookup_key=['dtRef', 'uf', 'tipo_cultura'],
    ),
    FeatureLookup(
        table_name=TABLE_FS_APOLICE_FINANCEIRO,
        lookup_key=['dtRef', 'apolice'],
    ),
    FeatureLookup(
        table_name=TABLE_FS_RISCO_SEGURADORA_CULTURA,
        lookup_key=['dtRef', 'seguradora', 'tipo_cultura'],
    ),
    FeatureLookup(
        table_name=TABLE_FS_ANOMALIA_TAXA,
        lookup_key=['dtRef', 'cultura', 'uf'],
    ),
    FeatureLookup(
        table_name=TABLE_FS_CONCENTRACAO_CARTEIRA,
        lookup_key=['dtRef', 'seguradora', 'mun'],
    ),
]

predict_set = fe.create_training_set(
    df=df_anchor,
    feature_lookups=feature_lookups,
    label=None
)

df_predict = predict_set.load_df().toPandas()
df_predict = derive_features(df_predict)

# COMMAND ----------

# DBTITLE 1,Inferência
probas = model.predict_proba(df_predict[model.feature_names_in_])

columns_id = ['dtRef', 'descModelName', 'nrModelVersion', 'apolice']

df_model = df_predict[['dtRef', 'apolice']].copy()
df_model['descModelName']  = model_name
df_model['nrModelVersion'] = actual_version
df_model['nrScore']        = probas[:, 1]           # prob(sinistro=1) para ranking
df_model[list(model.classes_)] = probas

df_long = (
    df_model
    .set_index(columns_id + ['nrScore'])
    .stack()
    .reset_index()
)
df_long.columns = columns_id + ['nrScore', 'descLabel', 'nrProbLabel']
df_long['descLabel'] = df_long['descLabel'].astype(int)

print(f'✓ Inferência concluída: {len(df_long):,} linhas ({len(df_long) // 2:,} apólices × 2 classes)')

# COMMAND ----------

# DBTITLE 1,Persistência
sdf = spark.createDataFrame(df_long)

# Idempotência — remove predições anteriores para o mesmo mês/modelo
if spark.catalog.tableExists(TABLE_PREDICOES):
    spark.sql(f"""
        DELETE FROM {TABLE_PREDICOES}
        WHERE dtRef = '{date}'
          AND descModelName = '{model_name}'
    """)
else:
    spark.sql(f"""
        CREATE TABLE {TABLE_PREDICOES} (
            dtRef DATE,
            descModelName STRING,
            nrModelVersion INT,
            apolice STRING,
            nrScore DOUBLE,
            descLabel INT,
            nrProbLabel DOUBLE
        )
        USING DELTA
        PARTITIONED BY (descModelName)
    """)

(
    sdf.write
    .format('delta')
    .mode('append')
    .partitionBy(['descModelName'])
    .saveAsTable(TABLE_PREDICOES)
)

print(f'✓ {len(df_long):,} predições salvas em {TABLE_PREDICOES}')
print(f'  dtRef={date}  model={model_name}  version={actual_version}')
