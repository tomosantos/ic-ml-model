# Databricks notebook source
# MAGIC %pip install -q databricks-feature-engineering mlflow xgboost
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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
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

sys.path.insert(0, '../model_sinistro')
from preprocessing import (
    FEATURES_CATEGORICAS,
    FEATURES_CICLICAS,
    FEATURES_NUMERICAS_APOLICE,
    FEATURES_NUMERICAS_HISTORICAS,
    derive_features,
    pipeline_tree,
)

mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment(experiment_id=2081689002426673)

# COMMAND ----------

# DBTITLE 1,Construção do Training Set
_sql_path    = os.path.join('../model_sinistro/', 'fl_sinistro.sql')
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
_exclude_columns = ['mun', 'uf', 'cultura', 'apolice']

training_set = fe.create_training_set(
    df=df_anchor,
    feature_lookups=feature_lookups,
    label='flSinistro',
    exclude_columns=_exclude_columns,
)

df = training_set.load_df().toPandas()
df = derive_features(df)

print(f'Dataset pré-filtro OOT: {len(df):,} linhas')

CUTOFF_OOT = pd.Timestamp('2025-01-01')
df['dtRef'] = pd.to_datetime(df['dtRef'])
df = df[df['dtRef'] < CUTOFF_OOT].copy()

print(f'Dataset pós-filtro OOT: {len(df):,} linhas (dtRef < {CUTOFF_OOT.date()})')
print(f'Maior dtRef pós-filtro: {df["dtRef"].max().date()}')

# COMMAND ----------

# DBTITLE 1,Split
LABEL    = 'flSinistro'
FEATURES = (
    FEATURES_NUMERICAS_HISTORICAS
    + FEATURES_NUMERICAS_APOLICE
    + FEATURES_CATEGORICAS
    + FEATURES_CICLICAS
)

_forbidden_features = {'dtRef', 'sinistro', 'indenizacao', 'sinistralidade', 'evento', 'flSinistro'}
_found_forbidden = _forbidden_features.intersection(FEATURES)
assert not _found_forbidden, f'Colunas proibidas em FEATURES: {_found_forbidden}'

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

# DBTITLE 1,Treinamento Inicial
def compute_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    y_pred   = (y_prob >= 0.5).astype(int)
    decil_10 = np.percentile(y_prob, 90)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc_roc':  roc_auc_score(y_true, y_prob),
        'auc_pr':   average_precision_score(y_true, y_prob),
        'ks':       ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0]).statistic,
        'f1':       f1_score(y_true, y_pred),
        'lift_10':  precision_score(y_true, (y_prob >= decil_10).astype(int)) / float(y_true.mean()),
    }


tree_models = {
    'decision_tree': (pipeline_tree, DecisionTreeClassifier(
        class_weight='balanced',
        max_depth=8,
        min_samples_leaf=50,
        random_state=42,
    )),
    'random_forest': (pipeline_tree, RandomForestClassifier(
        class_weight='balanced',
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=30,
        n_jobs=-1,
        random_state=42,
    )),
    'adaboost': (pipeline_tree, AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )),
    'xgboost': (pipeline_tree, XGBClassifier(
        scale_pos_weight=spw,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='auc',
        random_state=42,
    )),
}

param_grids = {
    'decision_tree': {
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [20, 50, 100],
        'min_samples_split': [2, 10, 20],
    },
    'random_forest': {
        'n_estimators': [150, 200, 250],
        'max_depth': [6, 8, 10, 12],
        'min_samples_leaf': [10, 20, 30],
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.03, 0.1, 0.3],
    },
    'xgboost': {
        'n_estimators': [150, 250],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
    },
}

baseline_results = {}

for name, (preproc_base, clf_base) in tree_models.items():
    preproc_run = clone(preproc_base)
    clf_seed    = clone(clf_base)
    pipeline    = Pipeline([('preproc', preproc_run), ('clf', clf_seed)])
    pipeline.fit(X_train, y_train)

    y_prob_train           = pipeline.predict_proba(X_train)[:, 1]
    y_prob_test            = pipeline.predict_proba(X_test)[:, 1]
    baseline_results[name] = {
        'pipeline': pipeline,
        'train':    compute_metrics(y_train, y_prob_train),
        'test':     compute_metrics(y_test,  y_prob_test),
    }
    print(
        f'✓ baseline_{name}  '
        f'auc_roc_train={baseline_results[name]["train"]["auc_roc"]:.4f}  '
        f'auc_roc_test={baseline_results[name]["test"]["auc_roc"]:.4f}'
    )

df_baseline = pd.DataFrame([
    {'model': name, **r['train'], **{f'{k}_test': v for k, v in r['test'].items()}}
    for name, r in baseline_results.items()
]).rename(columns={
    'accuracy': 'accuracy_train',
    'auc_roc': 'auc_roc_train',
    'auc_pr': 'auc_pr_train',
    'ks': 'ks_train',
    'f1': 'f1_train',
    'lift_10': 'lift_10_train',
})

champion_name = df_baseline.sort_values('auc_roc_test', ascending=False).iloc[0]['model']
print(f'\nCampeão inicial (baseline): {champion_name}')

# COMMAND ----------

# DBTITLE 1,Grid Search
champion_preproc, champion_clf = tree_models[champion_name]
grid_search = GridSearchCV(
    estimator=clone(champion_clf),
    param_grid=param_grids[champion_name],
    scoring='roc_auc',
    cv=3,
    refit=True,
    n_jobs=-1,
    verbose=4
)

tuned_pipeline = Pipeline([
    ('preproc', clone(champion_preproc)),
    ('grid', grid_search),
])
tuned_pipeline.fit(X_train, y_train)

grid_search = tuned_pipeline.named_steps['grid']

best_pipeline = tuned_pipeline
y_prob_train_tuned = best_pipeline.predict_proba(X_train)[:, 1]
y_prob_test_tuned = best_pipeline.predict_proba(X_test)[:, 1]
tuned_results = {
    'train': compute_metrics(y_train, y_prob_train_tuned),
    'test': compute_metrics(y_test, y_prob_test_tuned),
}

print(f'GridSearch concluído para {champion_name} | best_cv_auc_roc={grid_search.best_score_:.4f}')
print(f'Melhores hiperparâmetros: {grid_search.best_params_}')
print(
    f'✓ tuned_{champion_name}  '
    f'auc_roc_train={tuned_results["train"]["auc_roc"]:.4f}  '
    f'auc_roc_test={tuned_results["test"]["auc_roc"]:.4f}'
)

# COMMAND ----------

# DBTITLE 1,Quadro Comparativo
rows = []
for name, r in baseline_results.items():
    row = {'model': name}
    row.update({f'{k}_train': v for k, v in r['train'].items()})
    row.update({f'{k}_test':  v for k, v in r['test'].items()})
    rows.append(row)

df_results = pd.DataFrame(rows).sort_values('auc_roc_test', ascending=False)

print('\n── Comparativo de Modelos ───────────────────────────────────────────')
display(df_results.round(2))
print('─────────────────────────────────────────────────────────────────────\n')

# COMMAND ----------

# DBTITLE 1,Feature Importance — Champion Model
champion_estimator = best_pipeline.named_steps['grid'].best_estimator_
preproc_fitted = best_pipeline.named_steps['preproc']

if hasattr(preproc_fitted, 'get_feature_names_out'):
    feature_names = preproc_fitted.get_feature_names_out()
else:
    feature_names = np.array(FEATURES)

if hasattr(champion_estimator, 'feature_importances_'):
    importances = champion_estimator.feature_importances_
else:
    raise ValueError(f'Modelo campeão {champion_name} não expõe feature_importances_.')

if len(feature_names) != len(importances):
    feature_names = np.array([f'feature_{i}' for i in range(len(importances))])

df_feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances,
}).sort_values('importance', ascending=False)

print(f'\n── Feature Importance ({champion_name}) ─────────────────────────────')
display(df_feature_importance.round(6))
print('─────────────────────────────────────────────────────────────────────\n')

# COMMAND ----------

# DBTITLE 1,Registro do Campeão Tunado
print(f'Campeão baseline: {champion_name}')

with mlflow.start_run(run_name=champion_name) as run:
    mlflow.log_params({
        'sinistro_rate_train': sinistro_rate_train,
        'sinistro_rate_test':  sinistro_rate_test,
        'champion_baseline':   champion_name,
        'grid_cv':             3,
        'grid_refit':          True,
        'grid_scoring':        'roc_auc',
        'cutoff_oot':          str(CUTOFF_OOT.date()),
    })
    mlflow.log_metric('grid_best_cv_auc_roc', float(grid_search.best_score_))

    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    for k, v in baseline_results[champion_name]['train'].items():
        mlflow.log_metric(f'baseline_{k}_train', v)
    for k, v in baseline_results[champion_name]['test'].items():
        mlflow.log_metric(f'baseline_{k}_test', v)

    for k, v in tuned_results['train'].items():
        mlflow.log_metric(f'{k}_train', v)
    for k, v in tuned_results['test'].items():
        mlflow.log_metric(f'{k}_test', v)

    fe.log_model(
        model=best_pipeline,
        artifact_path='model',
        flavor=mlflow.xgboost,
        training_set=training_set,
        registered_model_name='04_feature_store.seg_rural.sinistro',
    )

print(f'✓ Modelo registrado: 04_feature_store.seg_rural.sinistro')
print(f'  auc_roc_train={tuned_results["train"]["auc_roc"]:.4f}  auc_roc_test={tuned_results["test"]["auc_roc"]:.4f}')
print('\n✓ Pipeline de treinamento concluído.')
