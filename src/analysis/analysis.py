# Databricks notebook source
# MAGIC %pip install -q matplotlib seaborn scikit-learn xgboost mlflow databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup
import glob as _glob
import os
import re as _re
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
PALETTE_MAIN     = '#2A788E'
PALETTE_NEG      = '#7E7E7E'
PALETTE_ACCENT   = '#FDE725'
PALETTE_LINE_CUM = '#D62728'


def _viridis(n: int) -> list:
    return [plt.cm.viridis(i / max(n - 1, 1)) for i in range(n)]



sns.set_theme(style='white', font='DejaVu Sans')
matplotlib.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
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
MLF_RUN_NAME  = '4c530133b755446e9466e3fdee71b5b8'


def save_fig(fig, name: str):
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=200)
    try:
        mlflow.log_artifact(path, artifact_path='figures')
    except Exception:
        pass
    display(fig)
    plt.close(fig)
    print(f'✓ figura salva: {path}')


_FI_NAME_MAP = {
    'AnoPlantio':              'Ano de Plantio',
    'ConcentracaoSeguradora':  'Conc. Seguradora',
    'SinistrosCulturaUf':      'Sinistros Cult./UF',
    'TaxaMediaCulturaUf':      'Taxa Média Cult./UF',
    'TaxaSinistroCulturaUf':   'Taxa Sinistro Cult./UF',
    'StdTaxaCulturaUf':        'Desvio Taxa Cult./UF',
    'ApolicesCulturaUf':       'Apólices Cult./UF',
    'ApolicesCulturaExata':    'Apólices Cult. Exata',
    'TaxaApolice':             'Taxa da Apólice',
    'SeveridadeSegCultura':    'Severidade Seg./Cult.',
    'RazaoCoberturaProd':      'Razão Cob./Prod.',
    'NivelCobMedioCulturaUf':  'Nível Cob. Cult./UF',
    'EventosDominante':        'Evento Dominante',
    'TaxaSinistroSegCultura':  'Taxa Sinistro Seg./Cult.',
    'ApolicesSegCultura':      'Apólices Seg./Cult.',
    'tipo_cultura':            'Tipo de Cultura',
    'seguradora':              'Seguradora',
    'regiao':                  'Região',
}

def _fmt_fi_name(fname: str) -> str:
    s = str(fname)
    s = _re.sub(r'^(num_nr|num_|cat_nr|cat_|cyc_sin_|cyc_cos_)', '', s)
    for key, val in _FI_NAME_MAP.items():
        if s.startswith(key):
            suffix = s[len(key):]
            m = _re.match(r'_?(\d+)d?', suffix)
            return f'{val} {m.group(1)}d' if m else val
    s = _re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.strip()

EVENTO_LABEL_MAP = {
    'seca':                      'Seca',
    'granizo':                   'Granizo',
    'outras':                    'Outros',
    'geada':                     'Geada',
    'chuva':                     'Chuva',
    'vento':                     'Vento',
    "INUNDACAO/TROMBA D'AGUA":   "Inundação/Tromba d'Água",
    'INUNDACAO/TROMBA D´AGUA':   "Inundação/Tromba d'Água",
    'INUNDACAO':                 'Inundação',
    'temp.':                     'Temperatura',
    'incendio':                  'Incêndio',
    'morte':                     'Morte Animal',
    'queda parr.':               'Queda Parcial',
    'raio':                      'Raio',
    'var. preco':                'Var. de Preço',
    'doencas':                   'Doenças',
    'perda qual.':               'Perda de Qualidade',
    'replantio':                 'Replantio',
}

def _fmt_evento(name: str) -> str:
    return EVENTO_LABEL_MAP.get(name, EVENTO_LABEL_MAP.get(name.strip(), name.title()))

CULTURA_LABEL_MAP = {
    'SOJA': 'Soja', 'MILHO': 'Milho', 'TRIGO': 'Trigo', 'ARROZ': 'Arroz',
    'SORGO': 'Sorgo', 'AVEIA': 'Aveia', 'CEVADA': 'Cevada', 'TRITICALE': 'Triticale',
    'ALGODAO': 'Algodão', 'CAFE': 'Café', 'CANA': 'Cana-de-açúcar', 'CANOLA': 'Canola',
    'GIRASSOL': 'Girassol', 'AMENDOIM': 'Amendoim', 'FEIJAO': 'Feijão',
    'UVA': 'Uva', 'MACA': 'Maçã', 'PERA': 'Pera', 'PESSEGO': 'Pêssego',
    'GRAOS': 'Grãos', 'FRUTAS': 'Frutas', 'HORTALICAS': 'Hortaliças',
    'FORRAGEIRA': 'Forrageiras', 'OUTROS': 'Outros', 'outras': 'Outros',
}

def _fmt_cultura(name: str) -> str:
    return CULTURA_LABEL_MAP.get(name, CULTURA_LABEL_MAP.get(name.upper(), name.title()))

# COMMAND ----------

# DBTITLE 1,[0] Carga da Silver Layer
df_silver = spark.table(TABLE_SILVER_CLEANED).toPandas()
df_silver['dt_inicio_vigencia'] = pd.to_datetime(df_silver['dt_inicio_vigencia'], errors='coerce')
df_silver['dt_fim_vigencia']    = pd.to_datetime(df_silver['dt_fim_vigencia'], errors='coerce')
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
df_anual = df_anual[df_anual['ano'] >= 2016].reset_index(drop=True)
anos = df_anual['ano'].astype(str).tolist()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

x = np.arange(len(anos))
bars = ax1.bar(x, df_anual['total_apolices'], color=PALETTE_MAIN,
               width=0.6, edgecolor='white', linewidth=0.5, zorder=3)
for bar, val in zip(bars, df_anual['total_apolices']):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
             f'{val:,.0f}', ha='center', va='bottom', fontsize=8, color='#1E293B')

ax2.plot(x, df_anual['taxa_sinistro'], color=PALETTE_ACCENT, linestyle='--',
         linewidth=2.5, marker='o', markersize=7,
         markeredgecolor='#444', markeredgewidth=1, zorder=4)
for i, val in enumerate(df_anual['taxa_sinistro']):
    ax2.text(i, val + 0.003, f'{val:.1%}', ha='center', va='bottom',
             fontsize=8.5, color='#997700')

ax1.set_xticks(x)
ax1.set_xticklabels(anos)
ax1.set_xlabel('Ano')
ax1.set_ylabel('Total de Apólices')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax2.set_ylabel('Taxa de Sinistro', color='#997700')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1%}'))
ax2.tick_params(axis='y', colors='#997700')
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.set_xlim(-0.5, len(x) - 0.5)

handles_11 = [
    Patch(color=PALETTE_MAIN, label='Apólices'),
    plt.Line2D([0], [0], color=PALETTE_ACCENT, linestyle='--', marker='o',
               markeredgecolor='#444', label='Taxa de Sinistro'),
]
ax1.legend(handles=handles_11, loc='upper left', framealpha=0.9)
ax1.set_title('Evolução do Volume de Apólices e Taxa de Sinistro (2016–2025)', pad=12)

plt.tight_layout()
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
df_regiao_taxa = df_regiao.sort_values('taxa', ascending=False).reset_index(drop=True)
n = len(df_regiao)
cum_reg = (df_regiao['total'].cumsum() / df_regiao['total'].sum() * 100).values

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

ax_l2 = ax_l.twinx()
x_r = np.arange(n)
bars_l = ax_l.bar(x_r, df_regiao['total'], color=_viridis(n),
                  width=0.6, edgecolor='white', linewidth=0.5, zorder=3)
for bar, val in zip(bars_l, df_regiao['total']):
    ax_l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
              f'{val:,.0f}', ha='center', va='bottom', fontsize=8)
ax_l2.plot(x_r, cum_reg, color=PALETTE_LINE_CUM, linestyle='--',
           linewidth=2, marker='o', markersize=6, zorder=4)
for i, val in enumerate(cum_reg):
    ax_l2.text(i, val + 1.5, f'{val:.0f}%', ha='center', va='bottom',
               fontsize=7.5, color=PALETTE_LINE_CUM)
ax_l.set_xticks(x_r)
ax_l.set_xticklabels(df_regiao['regiao'], rotation=15, ha='right')
ax_l.set_ylabel('Total de Apólices')
ax_l.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax_l2.set_ylabel('% Acumulado', color=PALETTE_LINE_CUM)
ax_l2.set_ylim(0, 115)
ax_l2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax_l2.tick_params(axis='y', colors=PALETTE_LINE_CUM)
ax_l.spines['top'].set_visible(False)
ax_l2.spines['top'].set_visible(False)
ax_l.set_title('Total de Apólices por Região', pad=10)

n_rt = len(df_regiao_taxa)
y_r = np.arange(n_rt)
bars_r = ax_r.barh(y_r, df_regiao_taxa['taxa'] * 100,
                   color=[plt.cm.Greens(0.9 - 0.6 * i / max(n_rt - 1, 1)) for i in range(n_rt)],
                   height=0.6, edgecolor='white', linewidth=0.5, zorder=3)
for bar, val in zip(bars_r, df_regiao_taxa['taxa'] * 100):
    ax_r.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
              f'{val:.1f}%', va='center', fontsize=8.5)
ax_r.set_yticks(y_r)
ax_r.set_yticklabels(df_regiao_taxa['regiao'])
ax_r.set_xlabel('Taxa de Sinistro (%)')
ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax_r.invert_yaxis()
ax_r.set_title('Taxa de Sinistro por Região', pad=10)
sns.despine(ax=ax_r)

fig.suptitle('Distribuição Geográfica do PSR (2016–2025)', y=1.02)
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
df_cultura['cultura_fmt'] = df_cultura['tipo_cultura'].map(_fmt_cultura)
df_cultura_taxa = df_cultura.sort_values('taxa', ascending=False).reset_index(drop=True)
taxa_media_global = df_silver['sinistro'].mean()
n_c = len(df_cultura)
cum_cult = (df_cultura['total'].cumsum() / df_cultura['total'].sum() * 100).values

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))

ax_l2 = ax_l.twinx()
x_c = np.arange(n_c)
bars_cl = ax_l.bar(x_c, df_cultura['total'], color=_viridis(n_c),
                   width=0.6, edgecolor='white', linewidth=0.5, zorder=3)
for bar, val in zip(bars_cl, df_cultura['total']):
    ax_l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
              f'{val:,.0f}', ha='center', va='bottom', fontsize=7)
ax_l2.plot(x_c, cum_cult, color=PALETTE_LINE_CUM, linestyle='--',
           linewidth=2, marker='o', markersize=5, zorder=4)
for i, val in enumerate(cum_cult):
    ax_l2.text(i, val + 1.5, f'{val:.0f}%', ha='center', va='bottom',
               fontsize=7, color=PALETTE_LINE_CUM)
ax_l.set_xticks(x_c)
ax_l.set_xticklabels(df_cultura['cultura_fmt'], rotation=35, ha='right', fontsize=8)
ax_l.set_xlim(-0.5, n_c - 0.5)
ax_l.set_ylabel('Total de Apólices')
ax_l.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax_l2.set_ylabel('% Acumulado', color=PALETTE_LINE_CUM)
ax_l2.set_ylim(0, 115)
ax_l2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax_l2.tick_params(axis='y', colors=PALETTE_LINE_CUM)
ax_l.spines['top'].set_visible(False)
ax_l2.spines['top'].set_visible(False)
ax_l.set_title('Total de Apólices por Tipo de Cultura', pad=10)

n_ct = len(df_cultura_taxa)
y_ct = np.arange(n_ct)
max_taxa_val = df_cultura_taxa['taxa'].max() * 100
bars_cr = ax_r.barh(y_ct, df_cultura_taxa['taxa'] * 100,
                    color=[plt.cm.Greens(0.9 - 0.6 * i / max(n_ct - 1, 1)) for i in range(n_ct)],
                    height=0.6, edgecolor='white', linewidth=0.5, zorder=3)
for bar, val in zip(bars_cr, df_cultura_taxa['taxa'] * 100):
    ax_r.text(val + max_taxa_val * 0.01, bar.get_y() + bar.get_height() / 2,
              f'{val:.1f}%', va='center', fontsize=8)
ax_r.axvline(x=taxa_media_global * 100, color=PALETTE_LINE_CUM, linestyle='--', linewidth=1.5)
ax_r.text(taxa_media_global * 100 + max_taxa_val * 0.01, 0.98,
          f'Média: {taxa_media_global:.1%}', color=PALETTE_LINE_CUM, fontsize=8.5,
          transform=ax_r.get_xaxis_transform(), va='top')
ax_r.set_xlim(0, max_taxa_val * 1.18)
ax_r.set_yticks(y_ct)
ax_r.set_yticklabels(df_cultura_taxa['cultura_fmt'], fontsize=8)
ax_r.set_xlabel('Taxa de Sinistro (%)')
ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax_r.invert_yaxis()
ax_r.set_title('Taxa de Sinistro por Tipo de Cultura', pad=10)
sns.despine(ax=ax_r)

fig.suptitle('Concentração por Tipo de Cultura — PSR (2016–2025)', y=1.02)
plt.tight_layout()
save_fig(fig, 'fig_1_3_cultura')

# COMMAND ----------

# DBTITLE 1,[4.1.4] Distribuição da variável resposta
sinistro_counts = df_silver['sinistro'].value_counts().sort_index()
taxa_sinistro   = df_silver['sinistro'].mean()
total           = len(df_silver)

labels_pie = [
    f'Sem Sinistro\n{sinistro_counts[0]:,.0f} ({sinistro_counts[0]/total:.1%})',
    f'Com Sinistro\n{sinistro_counts[1]:,.0f} ({sinistro_counts[1]/total:.1%})',
]

fig, ax = plt.subplots(figsize=(7, 6))
wedges, _ = ax.pie(
    sinistro_counts.values,
    labels=None,
    colors=[PALETTE_NEG, PALETTE_MAIN],
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2.5),
    startangle=90,
)
ax.legend(wedges, labels_pie, loc='lower center', ncol=2,
          bbox_to_anchor=(0.5, -0.08), fontsize=10, framealpha=0.9)
ax.set_title('Distribuição da Variável Resposta (flSinistro)', pad=15)
plt.tight_layout()
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

palette_box = ['Sem sinistro', 'Com sinistro']
color_box   = [PALETTE_NEG, PALETTE_MAIN]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for idx, var in enumerate(VARS_COMPARATIVO):
    ax = axes.flat[idx]
    subset = df_silver[[var, 'sinistro']].dropna().copy()
    subset['grupo'] = subset['sinistro'].map({0: 'Sem sinistro', 1: 'Com sinistro'})
    sns.boxplot(
        data=subset, x='grupo', y=var,
        order=palette_box,
        palette=dict(zip(palette_box, color_box)),
        width=0.5, linewidth=1.2, fliersize=2,
        ax=ax,
    )
    ax.set_xlabel('')
    ax.set_title(var)
    sns.despine(ax=ax)

handles_box = [
    Patch(color=PALETTE_NEG, label='Sem sinistro'),
    Patch(color=PALETTE_MAIN, label='Com sinistro'),
]
fig.legend(handles=handles_box, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
fig.suptitle('Distribuição das Variáveis Financeiras por Ocorrência de Sinistro', y=1.04)
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
df_eventos = df_eventos[df_eventos['pct'] > 0.001]
df_eventos['evento_fmt'] = df_eventos['evento'].map(_fmt_evento)

n_evt   = len(df_eventos)
cum_evt = (df_eventos['pct'].cumsum() * 100).values

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

x_e = np.arange(n_evt)
bars_e = ax1.bar(x_e, df_eventos['pct'] * 100, color=_viridis(n_evt),
                 width=0.6, edgecolor='white', linewidth=0.5, zorder=3)
for bar, val in zip(bars_e, df_eventos['pct'] * 100):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5)

ax2.plot(x_e, cum_evt, color=PALETTE_LINE_CUM, linestyle='--',
         linewidth=2, marker='o', markersize=6, zorder=4)
for i, val in enumerate(cum_evt):
    ax2.text(i, val + 1, f'{val:.0f}%', ha='center', va='bottom',
             fontsize=7.5, color=PALETTE_LINE_CUM)

ax1.set_xticks(x_e)
ax1.set_xticklabels(df_eventos['evento_fmt'], rotation=30, ha='right')
ax1.set_xlabel('Evento')
ax1.set_ylabel('Percentual (%)')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax2.set_ylabel('% Acumulado', color=PALETTE_LINE_CUM)
ax2.set_ylim(0, 115)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax2.tick_params(axis='y', colors=PALETTE_LINE_CUM)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.set_xlim(-0.5, n_evt - 0.5)
ax1.set_title('Eventos Causadores de Sinistro — PSR (2016–2025)', pad=12)
plt.tight_layout()
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
METRICS_LIST = ['accuracy', 'auc_roc', 'auc_pr', 'ks', 'f1', 'lift_10']
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
    for met in METRICS_LIST:
        row[f'{METRIC_LABELS[met]} (Treino)'] = metrics.get(
            f'baseline_{m}_{met}_train', metrics.get(f'baseline_{met}_train', np.nan)
        )
        row[f'{METRIC_LABELS[met]} (Teste)']  = metrics.get(
            f'baseline_{m}_{met}_test', metrics.get(f'baseline_{met}_test', np.nan)
        )
    rows.append(row)

df_baselines = pd.DataFrame(rows).sort_values('AUC-ROC (Teste)', ascending=False)
display(df_baselines.round(4))
df_baselines.to_csv(os.path.join(FIGURES_DIR, 'tab_2_1_baselines.csv'), index=False)

pivot_test = (
    pd.DataFrame([{
        'Modelo': MODEL_LABELS[m],
        **{METRIC_LABELS[met]: metrics.get(
            f'baseline_{m}_{met}_test', metrics.get(f'baseline_{met}_test', np.nan)
        ) for met in METRICS_LIST},
    } for m in MODELS])
    .set_index('Modelo')
    .sort_values('AUC-ROC', ascending=False)
)

fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(
    pivot_test, annot=True, fmt='.3f', cmap='viridis',
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Valor', 'shrink': 0.8},
    ax=ax,
)
ax.set_title('Desempenho dos Modelos Baseline — Conjunto de Teste', pad=12)
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
save_fig(fig, 'fig_2_1_baselines_heatmap')

# COMMAND ----------

# DBTITLE 1,[4.2.2] Baseline vs. Tuned Champion
champion = params.get('champion_baseline', 'xgboost')

df_comparison = pd.DataFrame({
    'Métrica':           [METRIC_LABELS[m] for m in METRICS_LIST],
    'Baseline (Treino)': [metrics.get(f'baseline_{champion}_{m}_train',
                           metrics.get(f'baseline_{m}_train', np.nan)) for m in METRICS_LIST],
    'Baseline (Teste)':  [metrics.get(f'baseline_{champion}_{m}_test',
                           metrics.get(f'baseline_{m}_test',  np.nan)) for m in METRICS_LIST],
    'Ajustado (Treino)': [metrics.get(f'{m}_train', np.nan) for m in METRICS_LIST],
    'Ajustado (Teste)':  [metrics.get(f'{m}_test',  np.nan) for m in METRICS_LIST],
})
display(df_comparison.round(4))
df_comparison.to_csv(os.path.join(FIGURES_DIR, 'tab_2_2_baseline_vs_tuned.csv'), index=False)

METRICS_PLOT = ['auc_roc', 'auc_pr', 'ks', 'lift_10']
labels_plot  = [METRIC_LABELS[m] for m in METRICS_PLOT]

baseline_vals = [metrics.get(f'baseline_{champion}_{m}_test',
                  metrics.get(f'baseline_{m}_test', np.nan)) for m in METRICS_PLOT]
tuned_vals    = [metrics.get(f'{m}_test', np.nan) for m in METRICS_PLOT]

vc = _viridis(2)
x_m = np.arange(len(labels_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x_m - width / 2, baseline_vals, width, color=vc[0],
               label='Baseline', edgecolor='white', linewidth=0.5, zorder=3)
bars2 = ax.bar(x_m + width / 2, tuned_vals, width, color=vc[1],
               label='Ajustado (GridSearchCV)', edgecolor='white', linewidth=0.5, zorder=3)

for bar, val in zip(bars1, baseline_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, tuned_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x_m)
ax.set_xticklabels(labels_plot)
ax.set_ylabel('Valor')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.3f}'))
ax.legend(framealpha=0.9)
ax.set_title(f'Comparativo Baseline vs. Modelo Ajustado — {MODEL_LABELS[champion]}', pad=12)
sns.despine(ax=ax)
plt.tight_layout()
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
df_predicoes = spark.table(TABLE_PREDICOES).toPandas()
df_predicoes['dtRef'] = pd.to_datetime(df_predicoes['dtRef'])

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
ax.plot(fpr_test, tpr_test, color=PALETTE_MAIN, linewidth=2.5,
        label=f'Teste (AUC = {auc_test:.3f})')
ax.fill_between(fpr_test, tpr_test, alpha=0.08, color=PALETTE_MAIN)
ax.plot([0, 1], [0, 1], color='#94A3B8', linewidth=1, linestyle=':', label='Aleatório')
ax.set_xlabel('Taxa de Falsos Positivos')
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1%}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1%}'))
ax.legend(loc='lower right', framealpha=0.9)
ax.set_title('Curva ROC — Conjunto de Teste', pad=12)
sns.despine(ax=ax)
plt.tight_layout()
save_fig(fig, 'fig_3_1_roc_curves')

# COMMAND ----------

# DBTITLE 1,[4.3.2] Curva Precision-Recall — Conjunto de Teste
prec_test, rec_test, _ = precision_recall_curve(y_test, y_prob_test)
ap_test = average_precision_score(y_test, y_prob_test)
taxa_sinistro_global = df_silver['sinistro'].mean()

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_test, prec_test, color=PALETTE_MAIN, linewidth=2.5,
        label=f'Teste (AP = {ap_test:.3f})')
ax.fill_between(rec_test, prec_test, alpha=0.08, color=PALETTE_MAIN)
ax.axhline(y=taxa_sinistro_global, color='#94A3B8', linestyle=':', linewidth=1.5,
           label=f'Baseline aleatório: {taxa_sinistro_global:.1%}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precisão')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1%}'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1%}'))
ax.legend(loc='upper right', framealpha=0.9)
ax.set_title('Curva Precision-Recall — Conjunto de Teste', pad=12)
sns.despine(ax=ax)
plt.tight_layout()
save_fig(fig, 'fig_3_2_pr_curves')

# COMMAND ----------

# DBTITLE 1,[4.3.3] Distribuição de probabilidades preditas por classe
scores_test  = y_prob_test
labels_test  = y_test.values
ks_stat_test = ks_2samp(scores_test[labels_test == 1], scores_test[labels_test == 0]).statistic
scores_oot   = df_oot[SCORE_COL].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(scores_test[labels_test == 0], bins=50, density=True,
         color=PALETTE_NEG, alpha=0.65, label='Sem sinistro',
         edgecolor='white', linewidth=0.3)
ax1.hist(scores_test[labels_test == 1], bins=50, density=True,
         color=PALETTE_MAIN, alpha=0.65, label='Com sinistro',
         edgecolor='white', linewidth=0.3)
ax1.set_xlabel('Probabilidade Predita de Sinistro')
ax1.set_ylabel('Densidade')
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax1.legend(framealpha=0.9)
ax1.set_title(f'Conjunto de Teste (KS = {ks_stat_test:.3f})')
sns.despine(ax=ax1)

ax2.hist(scores_oot, bins=50, density=True,
         color=PALETTE_MAIN, alpha=0.8, edgecolor='white', linewidth=0.3)
ax2.axvline(x=scores_oot.mean(), color=PALETTE_LINE_CUM, linestyle='--', linewidth=1.5)
ax2.text(scores_oot.mean() + 0.005, 0.95,
         f'Média: {scores_oot.mean():.3f}', color=PALETTE_LINE_CUM, fontsize=9,
         transform=ax2.get_xaxis_transform(), va='top')
ax2.set_xlabel('Probabilidade Predita de Sinistro')
ax2.set_ylabel('Densidade')
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax2.set_title(f'Período OOT — Monitoramento (n={len(scores_oot):,})')
sns.despine(ax=ax2)

fig.suptitle('Distribuição dos Escores Preditos', y=1.02)
plt.tight_layout()
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

def _assign_fi_group(f_str: str, base_map: dict) -> str:
    if f_str in base_map:
        return base_map[f_str]
    if 'SegCultura' in f_str:
        return 'Risco por Seguradora'
    if 'CulturaUf' in f_str or 'CulturaExata' in f_str:
        return 'Risco por Cultura/UF'
    if 'TaxaApolice' in f_str or 'Anomalia' in f_str:
        return 'Anomalia de Precificação'
    if 'Concentracao' in f_str or 'Seguradora' in f_str or 'HHI' in f_str:
        return 'Risco por Seguradora'
    if 'AnoPlantio' in f_str or 'Plantio' in f_str or 'Safra' in f_str:
        return 'Temporal / Safra'
    if 'EventosDominante' in f_str or 'EventoMun' in f_str:
        return 'Histórico por Município'
    if 'Cobertura' in f_str or 'Premio' in f_str or 'Area' in f_str:
        return 'Financeiras da Apólice'
    if 'Municipio' in f_str or 'Mun' in f_str:
        return 'Histórico por Município'
    if 'Taxa' in f_str and 'Sinistro' in f_str:
        return 'Risco por Cultura/UF'
    return 'Outros'

_base_groups = {}
for f in FEATURES_NUMERICAS_HISTORICAS:
    _base_groups[f] = 'Histórico por Município'
for f in FEATURES_NUMERICAS_APOLICE:
    _base_groups[f] = 'Financeiras da Apólice'
for f in FEATURES_CATEGORICAS:
    _base_groups[f] = 'Categóricas'
for f in FEATURES_CICLICAS:
    _base_groups[f] = 'Cíclicas (Sazonalidade)'

FEATURE_GROUPS = {str(f): _assign_fi_group(str(f), _base_groups) for f in feature_names}

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
df_top15_tbl = df_fi.head(15)[['feature', 'importance_pct', 'grupo']].copy()
df_top15_tbl.columns = ['Feature', 'Importância (%)', 'Grupo']
df_top15_tbl['Importância (%)'] = (df_top15_tbl['Importância (%)'] * 100).round(2)
df_top15_tbl.index = range(1, 16)
display(df_top15_tbl)
df_top15_tbl.to_csv(os.path.join(FIGURES_DIR, 'tab_4_1_top15_fi.csv'))
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
    'Temporal / Safra':         '#EF4444',
    'Outros':                   '#CBD5E1',
}

df_top15_fi = df_fi.head(15).copy()
df_top15_fi['feature_fmt'] = df_top15_fi['feature'].map(_fmt_fi_name)
cum_fi     = (df_top15_fi['importance_pct'].cumsum() * 100).values
bar_colors = df_top15_fi['grupo'].map(GROUP_COLORS).fillna('#CBD5E1').tolist()

fig, ax1 = plt.subplots(figsize=(13, 6))
ax2 = ax1.twinx()

x_fi = np.arange(len(df_top15_fi))
bars_fi = ax1.bar(x_fi, df_top15_fi['importance_pct'] * 100, color=bar_colors,
                  width=0.65, edgecolor='white', linewidth=0.5, zorder=3)
for bar, val in zip(bars_fi, df_top15_fi['importance_pct'] * 100):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

ax2.plot(x_fi, cum_fi, color=PALETTE_ACCENT, linestyle='--', linewidth=2.5,
         marker='o', markersize=7, markeredgecolor='#888', markeredgewidth=1, zorder=4)
for i, val in enumerate(cum_fi):
    ax2.text(i, val + 0.8, f'{val:.0f}%', ha='center', va='bottom',
             fontsize=7.5, color='#666')

ax1.set_xticks(x_fi)
ax1.set_xticklabels(df_top15_fi['feature_fmt'], rotation=35, ha='right', fontsize=8.5)
ax1.set_ylabel('Importância Relativa (%)')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax2.set_ylabel('% Acumulado (todas variáveis)', color='#555')
ax2.set_ylim(0, 115)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax2.tick_params(axis='y', colors='#555')
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.set_xlim(-0.5, len(x_fi) - 0.5)
ax1.set_title('Top-15 Variáveis por Importância — Modelo Campeão', pad=12)

shown_groups = set(df_top15_fi['grupo'].unique())
handles_fi = [Patch(color=col, label=grp) for grp, col in GROUP_COLORS.items() if grp in shown_groups]
handles_fi.append(plt.Line2D([0], [0], color=PALETTE_ACCENT, linestyle='--',
                              marker='o', markeredgecolor='#888', label='% Acumulado'))
ax1.legend(handles=handles_fi, title='Grupo', bbox_to_anchor=(1.12, 1),
           loc='upper left', fontsize=8, title_fontsize=9)

plt.tight_layout()
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

fig, ax = plt.subplots(figsize=(10, 5))
y_g = np.arange(len(df_group_imp))
colors_grp = [GROUP_COLORS.get(g, '#CBD5E1') for g in df_group_imp['Grupo']]
bars_g = ax.barh(y_g, df_group_imp['Importância Agregada (%)'], color=colors_grp,
                 height=0.6, edgecolor='white', linewidth=0.5, zorder=3)
for bar, val in zip(bars_g, df_group_imp['Importância Agregada (%)']):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=9)
ax.set_yticks(y_g)
ax.set_yticklabels(df_group_imp['Grupo'])
ax.set_xlabel('Importância Agregada (%)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
ax.invert_yaxis()
ax.set_title('Importância por Grupo de Features', pad=12)
sns.despine(ax=ax)
plt.tight_layout()
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
