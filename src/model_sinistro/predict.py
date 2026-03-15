# Databricks notebook source
# MAGIC %pip install -q databricks-feature-engineering mlflow xgboost
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup
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

# DBTITLE 1,Parâmetros
CUTOFF_OOT = pd.Timestamp('2025-01-01')
model_version = '9'

if not isinstance(CUTOFF_OOT, pd.Timestamp):
    raise ValueError('CUTOFF_OOT deve ser pandas.Timestamp')

if model_version != 'latest' and not model_version.isdigit():
    raise ValueError(f"model_version deve ser 'latest' ou um inteiro positivo. Recebido: '{model_version}'")

model_name = '04_feature_store.seg_rural.sinistro'

# COMMAND ----------

# DBTITLE 1,Carregar Modelo
model_uri    = f'models:/{model_name}/{model_version}'
model_pyfunc = mlflow.pyfunc.load_model(model_uri)
run_id       = model_pyfunc.metadata.run_id
model        = mlflow.sklearn.load_model(f'runs:/{run_id}/sklearn_pipeline')

if model_version == 'latest':
    versions       = MlflowClient().get_latest_versions(model_name)
    actual_version = int(versions[0].version)
else:
    actual_version = int(model_version)

print(f'✓ Modelo carregado: {model_name} v{actual_version}  (run_id={run_id})')

# COMMAND ----------

# DBTITLE 1,Âncora de Scoring
# Apólices OOT:
#   dtRef >= CUTOFF_OOT
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
    WHERE DATE_TRUNC('MONTH', dt_inicio_vigencia) >= DATE('{CUTOFF_OOT.date()}')
""")

count_anchor = df_anchor.count()
assert count_anchor > 0, f"Nenhuma apólice encontrada para dtRef >= {CUTOFF_OOT.date()}"
print(f'✓ Âncora: {count_anchor:,} apólices (dtRef >= {CUTOFF_OOT.date()})')

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

columns_id = ['dtRef', 'model_name', 'model_version', 'apolice']

df_model = df_predict[['dtRef', 'apolice']].copy()
df_model['model_name']  = model_name
df_model['model_version'] = actual_version
df_model['score']        = probas[:, 1]           # prob(sinistro=1) para ranking
df_model[list(model.classes_)] = probas

df_long = (
    df_model
    .set_index(columns_id + ['score'])
    .stack()
    .reset_index()
)
df_long.columns = columns_id + ['score', 'label', 'prob_label']
df_long['label'] = df_long['label'].astype(int)
df_long = df_long.drop(columns=['score'])

print(f'✓ Inferência concluída: {len(df_long):,} linhas ({len(df_long) // 2:,} apólices × 2 classes)')

# COMMAND ----------

# DBTITLE 1,Persistência
sdf = spark.createDataFrame(df_long)

# Full load — sobrescreve todas as predições para dtRef >= CUTOFF_OOT e infere schema
(
    sdf.write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .partitionBy(['model_name'])
    .saveAsTable(TABLE_PREDICOES)
)

print(f'✓ {len(df_long):,} predições salvas em {TABLE_PREDICOES}')
print(f'  dtRef>={CUTOFF_OOT.date()}  model={model_name}  version={actual_version}')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   t1.*,
# MAGIC   t2.sinistro
# MAGIC FROM `04_feature_store`.seg_rural.predicoes t1
# MAGIC LEFT JOIN `02_silver`.seg_rural.seg_cleaned t2
# MAGIC   ON t1.apolice = t2.apolice
# MAGIC   AND t1.dtRef = DATE_TRUNC('MONTH', t2.dt_inicio_vigencia)
