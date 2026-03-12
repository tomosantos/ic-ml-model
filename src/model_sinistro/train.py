# Databricks notebook source

# MAGIC %pip install databricks-feature-engineering mlflow xgboost
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup
import os
import sys

import mlflow
import numpy as np
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, '../lib')
from const import (
    TABLE_FS_ANOMALIA_TAXA,
    TABLE_FS_APOLICE_FINANCEIRO,
    TABLE_FS_CONCENTRACAO_CARTEIRA,
    TABLE_FS_HISTORICO_MUN,
    TABLE_FS_RISCO_CULTURA_UF,
    TABLE_FS_RISCO_SEGURADORA_CULTURA,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    FEATURES_CATEGORICAS,
    FEATURES_CICLICAS,
    FEATURES_NUMERICAS_APOLICE,
    FEATURES_NUMERICAS_HISTORICAS,
    derive_features,
    pipeline_linear,
    pipeline_tree,
)

mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment('/ic-ml-model/sinistro')

# COMMAND ----------

# DBTITLE 1,Construção do Training Set
_sql_path    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fl_sinistro.sql')
_anchor_sql  = open(_sql_path).read()
df_anchor    = spark.sql(_anchor_sql)

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

# Colunas de join puras — não são features preditoras
# tipo_cultura e seguradora são mantidos: usados como features categóricas em FEATURES_CATEGORICAS
_exclude_columns = ['mun', 'uf', 'cultura', 'dtRef', 'apolice']

training_set = fe.create_training_set(
    df=df_anchor,
    feature_lookups=feature_lookups,
    label='flSinistro',
    exclude_columns=_exclude_columns,
)

df = training_set.load_df().toPandas()
df = derive_features(df)

# COMMAND ----------

# DBTITLE 1,Split
LABEL    = 'flSinistro'
FEATURES = (
    FEATURES_NUMERICAS_HISTORICAS
    + FEATURES_NUMERICAS_APOLICE
    + FEATURES_CATEGORICAS
    + FEATURES_CICLICAS
)

X = df[FEATURES]
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)

sinistro_rate_train = float(y_train.mean())
sinistro_rate_test  = float(y_test.mean())
spw = float((y_train == 0).sum() / (y_train == 1).sum())

print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')
print(f'sinistro_rate  train={sinistro_rate_train:.4f}  test={sinistro_rate_test:.4f}')
print(f'scale_pos_weight: {spw:.2f}')

# COMMAND ----------

# DBTITLE 1,Treinamento dos Modelos
def compute_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    y_pred   = (y_prob >= 0.3).astype(int)
    decil_10 = np.percentile(y_prob, 90)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc_roc':  roc_auc_score(y_true, y_prob),
        'auc_pr':   average_precision_score(y_true, y_prob),
        'ks':       ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0]).statistic,
        'f1':       f1_score(y_true, y_pred),
        'lift_10':  precision_score(y_true, (y_prob >= decil_10).astype(int)) / float(y_true.mean()),
    }


sklearn_models = {
    'logistic_regression': (pipeline_linear, LogisticRegression(
        class_weight='balanced', max_iter=1000, C=0.1,
    )),
    'decision_tree': (pipeline_tree, DecisionTreeClassifier(
        class_weight='balanced', max_depth=6, min_samples_leaf=50,
    )),
    'random_forest': (pipeline_tree, RandomForestClassifier(
        class_weight='balanced', n_estimators=200,
        max_depth=8, min_samples_leaf=30, n_jobs=-1,
    )),
    'xgboost': (pipeline_tree, XGBClassifier(
        scale_pos_weight=spw, n_estimators=300,
        max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='aucpr',
    )),
}

results = {}

for name, (preproc_base, clf_base) in sklearn_models.items():
    preproc_run = clone(preproc_base)
    clf_run     = clone(clf_base)
    pipeline    = Pipeline([('preproc', preproc_run), ('clf', clf_run)])
    pipeline.fit(X_train, y_train)

    y_prob_train  = pipeline.predict_proba(X_train)[:, 1]
    y_prob_test   = pipeline.predict_proba(X_test)[:, 1]
    results[name] = {
        'pipeline': pipeline,
        'train':    compute_metrics(y_train, y_prob_train),
        'test':     compute_metrics(y_test,  y_prob_test),
    }
    print(f'✓ {name}  auc_pr_train={results[name]["train"]["auc_pr"]:.4f}  auc_pr_test={results[name]["test"]["auc_pr"]:.4f}')

# COMMAND ----------

# DBTITLE 1,Quadro Comparativo
rows = []
for name, r in results.items():
    row = {'model': name}
    row.update({f'{k}_train': v for k, v in r['train'].items()})
    row.update({f'{k}_test':  v for k, v in r['test'].items()})
    rows.append(row)

df_results = pd.DataFrame(rows).sort_values('auc_pr_test', ascending=False)

print('\n── Comparativo de Modelos ───────────────────────────────────────────')
print(df_results.to_string(index=False))
print('─────────────────────────────────────────────────────────────────────\n')

# COMMAND ----------

# DBTITLE 1,Registro do Campeão
champion_name  = df_results.iloc[0]['model']
best_pipeline  = results[champion_name]['pipeline']
print(f'Campeão: {champion_name}')

with mlflow.start_run(run_name=champion_name) as run:
    mlflow.log_params({
        'sinistro_rate_train': sinistro_rate_train,
        'sinistro_rate_test':  sinistro_rate_test,
        'champion':            champion_name,
    })
    for k, v in results[champion_name]['train'].items():
        mlflow.log_metric(f'{k}_train', v)
    for k, v in results[champion_name]['test'].items():
        mlflow.log_metric(f'{k}_test', v)

    fe.log_model(
        model=best_pipeline,
        artifact_path='model',
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name='feature_store.seg_rural.sinistro',
    )

client = mlflow.tracking.MlflowClient()
mv = client.get_latest_versions('feature_store.seg_rural.sinistro')[0]
client.set_registered_model_alias('feature_store.seg_rural.sinistro', 'Champion', mv.version)

print(f'✓ Alias "Champion" → feature_store.seg_rural.sinistro v{mv.version}')
print(f'  auc_pr_train={results[champion_name]["train"]["auc_pr"]:.4f}  auc_pr_test={results[champion_name]["test"]["auc_pr"]:.4f}')
print('\n✓ Pipeline de treinamento concluído.')
