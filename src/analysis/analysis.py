# Databricks notebook source
# MAGIC %pip install -q matplotlib seaborn scikit-learn xgboost mlflow databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup
import glob as _glob
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import mlflow
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
    f1_score, accuracy_score, precision_score,
)
from sklearn.model_selection import train_test_split
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

sys.path.insert(0, '../lib')
from const import (
    REGIAO_MAP,
    TABLE_SILVER_CLEANED,
    TABLE_PREDICOES,
    TABLE_FS_HISTORICO_MUN,
    TABLE_FS_RISCO_CULTURA_UF,
    TABLE_FS_APOLICE_FINANCEIRO,
    TABLE_FS_RISCO_SEGURADORA_CULTURA,
    TABLE_FS_ANOMALIA_TAXA,
    TABLE_FS_CONCENTRACAO_CARTEIRA,
)

sys.path.insert(0, '../model_sinistro')
from preprocessing import (
    FEATURES_CATEGORICAS, FEATURES_CICLICAS,
    FEATURES_NUMERICAS_APOLICE, FEATURES_NUMERICAS_HISTORICAS,
    derive_features,
)

# COMMAND ----------

# DBTITLE 1,Configurações globais
PALETTE_MAIN   = '#2563EB'
PALETTE_NEG    = '#64748B'
PALETTE_REGION = ['#1D4ED8', '#2563EB', '#3B82F6', '#60A5FA', '#93C5FD']
PALETTE_MODELS = ['#1E3A5F', '#2563EB', '#38BDF8', '#6EE7B7']

sns.set_theme(style='whitegrid', font='DejaVu Sans')
matplotlib.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
})

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    FIGURES_DIR = os.path.normpath(os.path.join(_nb_dir, '../../assets/analysis'))
except NameError:
    _nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    FIGURES_DIR = '/Workspace' + _nb_path.rsplit('/src/', 1)[0] + '/assets/analysis'

os.makedirs(FIGURES_DIR, exist_ok=True)

CUTOFF_OOT    = pd.Timestamp('2025-01-01')
LABEL_COL     = 'flSinistro'
SCORE_COL     = 'score'
MODEL_NAME    = '04_feature_store.seg_rural.sinistro'
EXPERIMENT_ID = 2081689002426673
MLF_RUN_NAME  = None


def save_fig(fig, name: str):
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    try:
        mlflow.log_artifact(path, artifact_path='figures')
    except Exception:
        pass
    plt.close(fig)
    print(f'✓ figura salva: {path}')

# COMMAND ----------

# DBTITLE 1,[0] Carga da Silver Layer
df_silver = spark.table(TABLE_SILVER_CLEANED).toPandas()
df_silver['dt_inicio_vigencia'] = pd.to_datetime(df_silver['dt_inicio_vigencia'])
df_silver['dt_fim_vigencia']    = pd.to_datetime(df_silver['dt_fim_vigencia'])
df_silver = df_silver[df_silver['dt_fim_vigencia'] < pd.Timestamp.today()].copy()
df_silver['ano']    = df_silver['dt_inicio_vigencia'].dt.year
df_silver['regiao'] = df_silver['uf'].map(REGIAO_MAP)

print(f'Silver: {len(df_silver):,} registros | sinistro={df_silver["sinistro"].mean():.4f}')
print(f'Anos: {df_silver["ano"].min()} – {df_silver["ano"].max()}')

# COMMAND ----------

# DBTITLE 1,[4.1.1] Volume de apólices por ano com taxa de sinistro
df_anual = (
    df_silver
    .groupby('ano')
    .agg(
        total_apolices=('apolice', 'count'),
        total_sinistros=('sinistro', 'sum'),
    )
    .assign(taxa_sinistro=lambda d: d['total_sinistros'] / d['total_apolices'])
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

x = df_anual['ano']
ax1.bar(x, df_anual['total_apolices'], color=PALETTE_MAIN, alpha=0.8, label='Apólices')
ax2.plot(x, df_anual['taxa_sinistro'], color='#DC2626', marker='o', linewidth=2, label='Taxa de sinistro')

ax1.set_xlabel('Ano')
ax1.set_ylabel('Total de Apólices')
ax2.set_ylabel('Taxa de Sinistro')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

ax1.set_title('Evolução do Volume de Apólices e Taxa de Sinistro (2016–2025)')
fig.text(0.5, -0.02, 'Fonte: SISSER/MAPA, 2026. Elaboração própria.', ha='center', fontsize=8, color='#64748B')
save_fig(fig, 'fig_1_1_volume_anual')
display(df_anual)

# COMMAND ----------

# DBTITLE 1,[4.1.2] Apólices e sinistros por região
df_regiao = (
    df_silver
    .groupby('regiao')
    .agg(
        total=('apolice', 'count'),
        sinistros=('sinistro', 'sum'),
    )
    .assign(taxa=lambda d: d['sinistros'] / d['total'])
    .sort_values('total', ascending=False)
    .reset_index()
)
df_regiao['pct_total'] = df_regiao['total'] / df_regiao['total'].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

regioes = df_regiao['regiao'].tolist()
n       = len(regioes)
cores   = PALETTE_REGION[:n] if n <= len(PALETTE_REGION) else PALETTE_REGION * (n // len(PALETTE_REGION) + 1)

container = ax1.barh(regioes, df_regiao['total'], color=cores[:n])
ax1.bar_label(container, fmt='{:,.0f}', padding=3)
ax1.set_xlabel('Total de Apólices')
ax1.set_title('Total de Apólices por Região')
ax1.invert_yaxis()

taxa_cores = [cores[i] for i in range(n)]
container2 = ax2.barh(regioes, df_regiao['taxa'] * 100, color=taxa_cores)
ax2.bar_label(container2, fmt='{:.1f}%', padding=3)
ax2.set_xlabel('Taxa de Sinistro (%)')
ax2.set_title('Taxa de Sinistro por Região (%)')
ax2.invert_yaxis()

fig.suptitle('Distribuição Geográfica do PSR (2016–2025)')
fig.text(0.5, -0.02, 'Fonte: SISSER/MAPA, 2026. Elaboração própria.', ha='center', fontsize=8, color='#64748B')
plt.tight_layout()
save_fig(fig, 'fig_1_2_distribuicao_regional')

tab_1_2 = df_regiao[['regiao', 'total', 'pct_total', 'sinistros', 'taxa']].copy()
tab_1_2.columns = ['Região', 'Apólices', '% do Total', 'Sinistros', 'Taxa (%)']
tab_1_2['% do Total'] = (tab_1_2['% do Total'] * 100).round(2)
tab_1_2['Taxa (%)']   = (tab_1_2['Taxa (%)'] * 100).round(2)
display(tab_1_2)

# COMMAND ----------

# DBTITLE 1,[4.1.3] Concentração por tipo de cultura
df_cultura = (
    df_silver
    .groupby('tipo_cultura')
    .agg(total=('apolice', 'count'), sinistros=('sinistro', 'sum'))
    .assign(
        pct_total=lambda d: d['total'] / d['total'].sum(),
        taxa=lambda d: d['sinistros'] / d['total'],
    )
    .sort_values('total', ascending=False)
    .reset_index()
)
taxa_media_global = df_silver['sinistro'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

culturas = df_cultura['tipo_cultura'].tolist()
n_c      = len(culturas)
cores_c  = (PALETTE_REGION * (n_c // len(PALETTE_REGION) + 1))[:n_c]

c1 = ax1.barh(culturas, df_cultura['total'], color=cores_c)
ax1.bar_label(c1, fmt='{:,.0f}', padding=3)
ax1.set_xlabel('Total de Apólices')
ax1.set_title('Total de Apólices por Tipo de Cultura')
ax1.invert_yaxis()

c2 = ax2.barh(culturas, df_cultura['taxa'] * 100, color=cores_c)
ax2.bar_label(c2, fmt='{:.1f}%', padding=3)
ax2.axvline(taxa_media_global * 100, color='#DC2626', linestyle='--', linewidth=1.5,
            label=f'Média geral: {taxa_media_global:.1%}')
ax2.set_xlabel('Taxa de Sinistro (%)')
ax2.set_title('Taxa de Sinistro por Tipo de Cultura (%)')
ax2.legend(fontsize=8)
ax2.invert_yaxis()

fig.suptitle('Concentração por Tipo de Cultura — PSR (2016–2025)')
fig.text(0.5, -0.02, 'Fonte: SISSER/MAPA, 2026. Elaboração própria.', ha='center', fontsize=8, color='#64748B')
plt.tight_layout()
save_fig(fig, 'fig_1_3_cultura')

# COMMAND ----------

# DBTITLE 1,[4.1.4] Distribuição da variável resposta
sinistro_counts = df_silver['sinistro'].value_counts().sort_index()
taxa_sinistro   = df_silver['sinistro'].mean()
total           = len(df_silver)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ['Sem sinistro', 'Com sinistro'],
    sinistro_counts.values,
    color=[PALETTE_NEG, PALETTE_MAIN],
    width=0.5,
)

for bar, count in zip(bars, sinistro_counts.values):
    pct = count / total
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.005,
        f'{count:,.0f} ({pct:.2%})',
        ha='center', va='bottom', fontsize=10,
    )

ax.axhline(total * taxa_sinistro, color='#DC2626', linestyle='--', linewidth=1.5)
ax.annotate(
    f'Taxa de sinistro: {taxa_sinistro:.2%}',
    xy=(1, total * taxa_sinistro),
    xytext=(0.6, total * taxa_sinistro + total * 0.02),
    fontsize=9, color='#DC2626',
)

ax.set_ylabel('Número de Apólices')
ax.set_title('Distribuição da Variável Resposta (flSinistro)')
fig.text(0.5, -0.02, 'Fonte: SISSER/MAPA, 2026. Elaboração própria.', ha='center', fontsize=8, color='#64748B')
save_fig(fig, 'fig_1_4_target_distribution')

# COMMAND ----------

# DBTITLE 1,[4.1.5] Resumo estatístico — variáveis financeiras
VARS_FINANCEIRAS = ['area', 'premio', 'taxa', 'subvencao', 'total_seg', 'nivel_cob']

df_resumo = (
    df_silver[VARS_FINANCEIRAS]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .loc[['mean', 'min', '25%', '50%', '75%', 'max', 'std']]
    .T
    .rename(columns={
        'mean': 'Média', 'min': 'Mínimo', '25%': 'Q1',
        '50%': 'Mediana', '75%': 'Q3', 'max': 'Máximo', 'std': 'Desvio Padrão',
    })
)
display(df_resumo.round(2))
df_resumo.round(2).to_csv(os.path.join(FIGURES_DIR, 'tab_1_5_resumo_estatistico.csv'))
print('✓ tab_1_5_resumo_estatistico.csv salvo')

# COMMAND ----------

# DBTITLE 1,[4.1.6] Comparativo sinistro vs. sem sinistro
VARS_COMPARATIVO = ['area', 'premio', 'taxa', 'subvencao', 'total_seg', 'nivel_cob']

df_comparativo = (
    df_silver
    .groupby('sinistro')[VARS_COMPARATIVO]
    .agg(['mean', 'median'])
)
df_comparativo.index = ['Sem sinistro', 'Com sinistro']
display(df_comparativo.round(2))

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes_flat = axes.flatten()

for ax, var in zip(axes_flat, VARS_COMPARATIVO):
    sns.boxplot(
        data=df_silver, x='sinistro', y=var,
        palette=[PALETTE_NEG, PALETTE_MAIN],
        showfliers=False, ax=ax,
    )
    ax.set_xticklabels(['Sem sinistro', 'Com sinistro'])
    ax.set_xlabel('')
    ax.set_title(var)

fig.suptitle('Distribuição das Variáveis Financeiras por Ocorrência de Sinistro')
fig.text(0.5, -0.01, 'Fonte: SISSER/MAPA, 2026. Elaboração própria.', ha='center', fontsize=8, color='#64748B')
plt.tight_layout()
save_fig(fig, 'fig_1_6_comparativo_grupos')

# COMMAND ----------

# DBTITLE 1,[4.1.7] Eventos causadores de sinistro
df_eventos = (
    df_silver[df_silver['sinistro'] == 1]
    .groupby('evento')
    .agg(total=('apolice', 'count'))
    .assign(pct=lambda d: d['total'] / d['total'].sum())
    .sort_values('total', ascending=False)
    .reset_index()
)
df_eventos = df_eventos[df_eventos['evento'] != 'nenhum']

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(df_eventos['evento'], df_eventos['pct'] * 100, color=PALETTE_MAIN, alpha=0.85)

for i, (_, row) in enumerate(df_eventos.iterrows()):
    ax.text(row['pct'] * 100 + 0.2, i, f"{row['pct']:.1%}", va='center', fontsize=9)

ax.invert_yaxis()
ax.set_xlabel('Percentual (%)')
ax.set_title('Eventos Causadores de Sinistro — PSR (2016–2025)')
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
fig.text(0.5, -0.02, 'Fonte: SISSER/MAPA, 2026. Elaboração própria.', ha='center', fontsize=8, color='#64748B')
save_fig(fig, 'fig_1_7_eventos')

# COMMAND ----------

# DBTITLE 1,[0] Carga dos resultados do MLflow
mlflow.set_registry_uri('databricks-uc')
client = mlflow.tracking.MlflowClient()

runs = client.search_runs(
    experiment_ids=[str(EXPERIMENT_ID)],
    filter_string="status = 'FINISHED'",
    order_by=['start_time DESC'],
    max_results=10,
)

run     = runs[0]
run_id  = run.info.run_id
params  = run.data.params
metrics = run.data.metrics

print(f'Run ID: {run_id}')
print(f'Campeão: {params.get("champion_baseline")}')
print(f'Cutoff OOT: {params.get("cutoff_oot")}')

# COMMAND ----------

# DBTITLE 1,[4.2.1] Tabela de baselines
MODELS = ['decision_tree', 'random_forest', 'adaboost', 'xgboost']
METRICS = ['accuracy', 'auc_roc', 'auc_pr', 'ks', 'f1', 'lift_10']
METRIC_LABELS = {
    'accuracy': 'Acurácia',
    'auc_roc':  'AUC-ROC',
    'auc_pr':   'AUC-PR',
    'ks':       'KS',
    'f1':       'F1',
    'lift_10':  'Lift@10%',
}
MODEL_LABELS = {
    'decision_tree': 'Árvore de Decisão',
    'random_forest': 'Random Forest',
    'adaboost':      'AdaBoost',
    'xgboost':       'XGBoost',
}

rows = []
for m in MODELS:
    row = {'Modelo': MODEL_LABELS[m]}
    for met in METRICS:
        row[f'{METRIC_LABELS[met]} (Treino)'] = metrics.get(f'baseline_{met}_train', np.nan)
        row[f'{METRIC_LABELS[met]} (Teste)']  = metrics.get(f'baseline_{met}_test',  np.nan)
    rows.append(row)

df_baselines = pd.DataFrame(rows).sort_values('AUC-ROC (Teste)', ascending=False)
display(df_baselines.round(4))
df_baselines.to_csv(os.path.join(FIGURES_DIR, 'tab_2_1_baselines.csv'), index=False)

# Heatmap — métricas de teste
pivot_test = (
    pd.DataFrame([{
        'Modelo': MODEL_LABELS[m],
        **{METRIC_LABELS[met]: metrics.get(f'baseline_{met}_test', np.nan) for met in METRICS},
    } for m in MODELS])
    .set_index('Modelo')
    .sort_values('AUC-ROC', ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(pivot_test, annot=True, fmt='.3f', cmap='Blues', linewidths=0.5,
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Desempenho dos Modelos Baseline — Conjunto de Teste')
ax.set_ylabel('')
plt.tight_layout()
save_fig(fig, 'fig_2_1_baselines_heatmap')

# COMMAND ----------

# DBTITLE 1,[4.2.2] Baseline vs. Tuned Champion
champion = params.get('champion_baseline', 'xgboost')

df_comparison = pd.DataFrame({
    'Métrica':          [METRIC_LABELS[m] for m in METRICS],
    'Baseline (Treino)': [metrics.get(f'baseline_{m}_train', np.nan) for m in METRICS],
    'Baseline (Teste)':  [metrics.get(f'baseline_{m}_test',  np.nan) for m in METRICS],
    'Ajustado (Treino)': [metrics.get(f'{m}_train', np.nan) for m in METRICS],
    'Ajustado (Teste)':  [metrics.get(f'{m}_test',  np.nan) for m in METRICS],
})
display(df_comparison.round(4))
df_comparison.to_csv(os.path.join(FIGURES_DIR, 'tab_2_2_baseline_vs_tuned.csv'), index=False)

METRICS_PLOT = ['auc_roc', 'auc_pr', 'ks', 'lift_10']
labels_plot  = [METRIC_LABELS[m] for m in METRICS_PLOT]

baseline_vals = [metrics.get(f'baseline_{m}_test', np.nan) for m in METRICS_PLOT]
tuned_vals    = [metrics.get(f'{m}_test',           np.nan) for m in METRICS_PLOT]

x_pos = np.arange(len(METRICS_PLOT))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x_pos - width / 2, baseline_vals, width, label='Baseline',                color='#93C5FD')
ax.bar(x_pos + width / 2, tuned_vals,    width, label='Ajustado (GridSearchCV)',  color=PALETTE_MAIN)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_plot)
ax.set_ylabel('Valor')
ax.set_title(f'Comparativo Baseline vs. Modelo Ajustado — {MODEL_LABELS[champion]}')
ax.legend()
fig.text(0.5, -0.02, 'Fonte: SISSER/MAPA, 2026. Elaboração própria.', ha='center', fontsize=8, color='#64748B')
save_fig(fig, 'fig_2_2_comparison_bar')

# COMMAND ----------

# DBTITLE 1,[4.2.3] Melhores hiperparâmetros
HYPERPARAM_KEYS = ['n_estimators', 'max_depth', 'learning_rate',
                   'min_samples_leaf', 'subsample', 'colsample_bytree']

hp_rows = {k: params[k] for k in HYPERPARAM_KEYS if k in params}
df_hyperparams = pd.DataFrame(list(hp_rows.items()), columns=['Hiperparâmetro', 'Valor'])
df_hyperparams['CV AUC-ROC'] = metrics.get('grid_best_cv_auc_roc', np.nan)
display(df_hyperparams)

# COMMAND ----------

# DBTITLE 1,[0] Carga do modelo campeão e conjunto de predições
# — Conjunto OOT: escores apenas (apólices 2025+ ainda não maturadas — sinistro indisponível) —
df_predicoes = spark.table(TABLE_PREDICOES).toPandas()
df_predicoes['dtRef'] = pd.to_datetime(df_predicoes['dtRef'])

# Tabela em formato long — filtrar label=1 para obter prob(sinistro=1) por apólice
df_oot = (
    df_predicoes[
        (df_predicoes['label'] == 1) &
        (df_predicoes['dtRef'] >= CUTOFF_OOT)
    ][['dtRef', 'apolice', 'prob_label']]
    .rename(columns={'prob_label': SCORE_COL})
    .copy()
)

assert SCORE_COL in df_oot.columns, f"Coluna '{SCORE_COL}' não encontrada em df_oot"
assert len(df_oot) > 0, 'df_oot vazio — verificar TABLE_PREDICOES e período OOT'

print(f'OOT: {len(df_oot):,} apólices (sem rótulos — ciclo não encerrado em 2025)')

# — Conjunto de Teste in-sample: recriar split idêntico ao train.py —
_sql_path = os.path.join('../model_sinistro/', 'fl_sinistro.sql')
_anchor_sql = open(_sql_path).read()
df_anchor_train = spark.sql(_anchor_sql)

fe_client = FeatureEngineeringClient()
feature_lookups = [
    FeatureLookup(table_name=TABLE_FS_HISTORICO_MUN,            lookup_key=['dtRef', 'mun']),
    FeatureLookup(table_name=TABLE_FS_RISCO_CULTURA_UF,         lookup_key=['dtRef', 'uf', 'tipo_cultura']),
    FeatureLookup(table_name=TABLE_FS_APOLICE_FINANCEIRO,       lookup_key=['dtRef', 'apolice']),
    FeatureLookup(table_name=TABLE_FS_RISCO_SEGURADORA_CULTURA, lookup_key=['dtRef', 'seguradora', 'tipo_cultura']),
    FeatureLookup(table_name=TABLE_FS_ANOMALIA_TAXA,            lookup_key=['dtRef', 'cultura', 'uf']),
    FeatureLookup(table_name=TABLE_FS_CONCENTRACAO_CARTEIRA,    lookup_key=['dtRef', 'seguradora', 'mun']),
]

training_set = fe_client.create_training_set(
    df=df_anchor_train,
    feature_lookups=feature_lookups,
    label=LABEL_COL,
    exclude_columns=['mun', 'uf', 'cultura', 'apolice'],
)
df_train_full = training_set.load_df().toPandas()
df_train_full = derive_features(df_train_full)
df_train_full['dtRef'] = pd.to_datetime(df_train_full['dtRef'])
df_train_full = df_train_full[df_train_full['dtRef'] < CUTOFF_OOT].copy()

FEATURES = (FEATURES_NUMERICAS_HISTORICAS + FEATURES_NUMERICAS_APOLICE
            + FEATURES_CATEGORICAS + FEATURES_CICLICAS)

X = df_train_full[FEATURES]
y = df_train_full[LABEL_COL]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

loaded_model = mlflow.sklearn.load_model(f'runs:/{run_id}/sklearn_pipeline')
y_prob_test  = loaded_model.predict_proba(X_test)[:, 1]

print(f'Teste (in-sample): {len(X_test):,} | OOT (escores): {len(df_oot):,}')

# COMMAND ----------

# DBTITLE 1,[4.3.1] Curva ROC — Conjunto de Teste
fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
auc_test = roc_auc_score(y_test, y_prob_test)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_test, tpr_test, color=PALETTE_MAIN, lw=2, label=f'Teste (AUC = {auc_test:.3f})')
ax.plot([0, 1], [0, 1], color='#94A3B8', lw=1, linestyle=':')

ax.set_xlabel('Taxa de Falsos Positivos')
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
ax.set_title('Curva ROC — Conjunto de Teste')
ax.legend(loc='lower right')
ax.annotate(
    'Regressão logística simples\n(Garcia, 2023): AUC = 0,56',
    xy=(0.65, 0.4), fontsize=8, color='#64748B', style='italic',
)
save_fig(fig, 'fig_3_1_roc_curves')

# COMMAND ----------

# DBTITLE 1,[4.3.2] Curva Precision-Recall — Conjunto de Teste
prec_test, rec_test, _ = precision_recall_curve(y_test, y_prob_test)
ap_test = average_precision_score(y_test, y_prob_test)

taxa_sinistro_global = df_silver['sinistro'].mean()

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_test, prec_test, color=PALETTE_MAIN, lw=2, label=f'Teste (AP = {ap_test:.3f})')
ax.axhline(taxa_sinistro_global, color='#94A3B8', lw=1.5, linestyle=':',
           label=f'Baseline aleatório: {taxa_sinistro_global:.1%}')
ax.annotate(
    f'Baseline aleatório: {taxa_sinistro_global:.1%}',
    xy=(0.7, taxa_sinistro_global + 0.01), fontsize=8, color='#64748B',
)

ax.set_xlabel('Recall')
ax.set_ylabel('Precisão')
ax.set_title('Curva Precision-Recall — Conjunto de Teste')
ax.legend(loc='upper right')
save_fig(fig, 'fig_3_2_pr_curves')

# COMMAND ----------

# DBTITLE 1,[4.3.3] Distribuição de probabilidades preditas por classe
# Painel esquerdo: teste (com separação por classe) | Painel direito: OOT (monitoramento — sem rótulos)
fig, (ax_test, ax_oot) = plt.subplots(1, 2, figsize=(13, 5))

scores_test  = y_prob_test
labels_test  = y_test.values
ks_stat_test = ks_2samp(scores_test[labels_test == 1], scores_test[labels_test == 0]).statistic

ax_test.hist(scores_test[labels_test == 0], bins=50, alpha=0.65, color=PALETTE_NEG,
             label='Sem sinistro', density=True)
ax_test.hist(scores_test[labels_test == 1], bins=50, alpha=0.65, color=PALETTE_MAIN,
             label='Com sinistro', density=True)
ax_test.set_title(f'Conjunto de Teste (KS = {ks_stat_test:.3f})')
ax_test.set_xlabel('Probabilidade Predita de Sinistro')
ax_test.set_ylabel('Densidade')
ax_test.legend()

scores_oot = df_oot[SCORE_COL].values
ax_oot.hist(scores_oot, bins=50, alpha=0.8, color=PALETTE_MAIN, density=True)
ax_oot.axvline(scores_oot.mean(), color='#DC2626', linestyle='--', linewidth=1.5,
               label=f'Média: {scores_oot.mean():.3f}')
ax_oot.set_title(f'Período OOT — Monitoramento de Escores (n={len(scores_oot):,})')
ax_oot.set_xlabel('Probabilidade Predita de Sinistro')
ax_oot.set_ylabel('Densidade')
ax_oot.legend()
ax_oot.text(0.98, 0.95, 'Rótulos indisponíveis\n(ciclo não encerrado)',
            transform=ax_oot.transAxes, ha='right', va='top',
            fontsize=8, color='#64748B', style='italic')

fig.suptitle('Distribuição dos Escores Preditos', fontsize=13)
save_fig(fig, 'fig_3_3_score_distribution')

# COMMAND ----------

# DBTITLE 1,[4.3.4] Tabela de métricas — Conjunto de Teste
def compute_metrics_full(y_true, y_prob):
    y_pred   = (y_prob >= 0.5).astype(int)
    decil_10 = np.percentile(y_prob, 90)
    return {
        'AUC-ROC':  roc_auc_score(y_true, y_prob),
        'AUC-PR':   average_precision_score(y_true, y_prob),
        'KS':       ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0]).statistic,
        'F1':       f1_score(y_true, y_pred),
        'Lift@10%': precision_score(y_true, (y_prob >= decil_10).astype(int)) / float(y_true.mean()),
        'Acurácia': accuracy_score(y_true, y_pred),
    }


metrics_test = compute_metrics_full(y_test.values, y_prob_test)

df_metrics = pd.DataFrame({
    'Métrica': list(metrics_test.keys()),
    'Teste':   list(metrics_test.values()),
    'OOT (2025)': ['N/D'] * len(metrics_test),
})
df_metrics.attrs['nota'] = 'OOT indisponível: apólices 2025 ainda não maturadas (ciclo não encerrado).'
display(df_metrics.assign(Teste=df_metrics['Teste'].round(4)))
df_metrics.to_csv(os.path.join(FIGURES_DIR, 'tab_3_4_metrics_test.csv'), index=False)
print('✓ tab_3_4_metrics_test.csv salvo')
print('  Nota: OOT indisponível — apólices 2025 ainda não maturadas.')

# COMMAND ----------

# DBTITLE 1,[0] Extração de Feature Importance do modelo campeão
champion_estimator = loaded_model.named_steps['grid'].best_estimator_
preproc_fitted     = loaded_model.named_steps['preproc']

FEATURES_ALL = (FEATURES_NUMERICAS_HISTORICAS + FEATURES_NUMERICAS_APOLICE
                + FEATURES_CATEGORICAS + FEATURES_CICLICAS)

if hasattr(preproc_fitted, 'get_feature_names_out'):
    feature_names = preproc_fitted.get_feature_names_out()
else:
    feature_names = np.array(FEATURES_ALL)

importances = champion_estimator.feature_importances_

if len(feature_names) != len(importances):
    feature_names = np.array([f'feature_{i}' for i in range(len(importances))])

FEATURE_GROUPS = {}
for f in FEATURES_NUMERICAS_HISTORICAS:
    FEATURE_GROUPS[f] = 'Histórico por Município'
for f in FEATURES_NUMERICAS_APOLICE:
    FEATURE_GROUPS[f] = 'Financeiras da Apólice'
for f in FEATURES_CATEGORICAS:
    FEATURE_GROUPS[f] = 'Categóricas'
for f in FEATURES_CICLICAS:
    FEATURE_GROUPS[f] = 'Cíclicas (Sazonalidade)'
for f in feature_names:
    f_str = str(f)
    if 'SegCultura' in f_str or 'Seguradora' in f_str or 'HHI' in f_str or 'Pct' in f_str:
        FEATURE_GROUPS[f_str] = 'Risco por Seguradora'
    elif 'CulturaUf' in f_str:
        FEATURE_GROUPS[f_str] = 'Risco por Cultura/UF'
    elif 'Anomalia' in f_str or 'Taxa' in f_str:
        FEATURE_GROUPS[f_str] = 'Anomalia de Precificação'

df_fi = (
    pd.DataFrame({'feature': feature_names, 'importance': importances})
    .sort_values('importance', ascending=False)
    .reset_index(drop=True)
)
df_fi['grupo']          = df_fi['feature'].map(lambda x: FEATURE_GROUPS.get(str(x), 'Outros'))
df_fi['importance_pct'] = df_fi['importance'] / df_fi['importance'].sum()

print(f'Top-15 features:\n{df_fi.head(15).to_string(index=False)}')
df_fi.to_csv(os.path.join(FIGURES_DIR, 'tab_4_fi_full.csv'), index=False)

# COMMAND ----------

# DBTITLE 1,[4.4.1] Tabela Top-15 Feature Importance
df_top15 = df_fi.head(15)[['feature', 'importance_pct', 'grupo']].copy()
df_top15.columns = ['Feature', 'Importância (%)', 'Grupo']
df_top15['Importância (%)'] = (df_top15['Importância (%)'] * 100).round(2)
df_top15.index = range(1, 16)
display(df_top15)
df_top15.to_csv(os.path.join(FIGURES_DIR, 'tab_4_1_top15_fi.csv'))
print('✓ tab_4_1_top15_fi.csv salvo')

# COMMAND ----------

# DBTITLE 1,[4.4.2] Gráfico de Feature Importance
GROUP_COLORS = {
    'Histórico por Município':  '#1D4ED8',
    'Risco por Cultura/UF':     '#2563EB',
    'Risco por Seguradora':     '#3B82F6',
    'Anomalia de Precificação': '#F59E0B',
    'Financeiras da Apólice':   '#10B981',
    'Categóricas':              '#8B5CF6',
    'Cíclicas (Sazonalidade)':  '#64748B',
    'Outros':                   '#CBD5E1',
}

df_top20 = df_fi.head(20).copy()
colors   = df_top20['grupo'].map(GROUP_COLORS).fillna('#CBD5E1')

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    y=range(len(df_top20)),
    width=df_top20['importance_pct'],
    color=colors,
    edgecolor='white',
    linewidth=0.5,
)
ax.set_yticks(range(len(df_top20)))
ax.set_yticklabels(df_top20['feature'], fontsize=9)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_xlabel('Importância Relativa')
ax.set_title('Top-20 Variáveis por Importância — Modelo Campeão')

legend_handles = [
    Patch(color=v, label=k)
    for k, v in GROUP_COLORS.items()
    if k in df_top20['grupo'].values
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8, title='Grupo')

save_fig(fig, 'fig_4_2_feature_importance')

# COMMAND ----------

# DBTITLE 1,[4.4.3] Importância por grupo de features
df_group_imp = (
    df_fi
    .groupby('grupo')['importance_pct']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
df_group_imp.columns = ['Grupo', 'Importância Agregada (%)']
df_group_imp['Importância Agregada (%)'] = (df_group_imp['Importância Agregada (%)'] * 100).round(2)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(
    df_group_imp['Grupo'],
    df_group_imp['Importância Agregada (%)'],
    color=[GROUP_COLORS.get(g, '#CBD5E1') for g in df_group_imp['Grupo']],
)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xlabel('Importância Agregada (%)')
ax.set_title('Importância por Grupo de Features')

for i, (_, row) in enumerate(df_group_imp.iterrows()):
    ax.text(
        row['Importância Agregada (%)'] + 0.2, i,
        f"{row['Importância Agregada (%)']:.1f}%",
        va='center', fontsize=9,
    )

save_fig(fig, 'fig_4_3_importance_by_group')
display(df_group_imp)

# COMMAND ----------

# DBTITLE 1,[5] Sumário de todos os artefatos gerados
figures = sorted(_glob.glob(os.path.join(FIGURES_DIR, '*.png')))
csvs    = sorted(_glob.glob(os.path.join(FIGURES_DIR, '*.csv')))

print(f'\n{"=" * 60}')
print(f'ARTEFATOS GERADOS — {len(figures)} figuras | {len(csvs)} tabelas')
print('=' * 60)
print('\nFiguras:')
for f in figures:
    print(f'  {os.path.basename(f)}')
print('\nTabelas CSV:')
for f in csvs:
    print(f'  {os.path.basename(f)}')
print('\nDiretório:', FIGURES_DIR)
print('=' * 60)
