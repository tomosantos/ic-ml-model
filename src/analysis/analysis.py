# Databricks notebook source
# MAGIC %pip install -q matplotlib seaborn scikit-learn xgboost mlflow databricks-feature-engineering plotly "kaleido==0.2.1"
dbutils.library.restartPython()

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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
PLOTLY_TEMPLATE = 'plotly_white'

def _viridis(n: int) -> list:
    return px.colors.sample_colorscale('Viridis', [i / max(n - 1, 1) for i in range(n)])

def _greens(n: int) -> list:
    return px.colors.sample_colorscale('Greens', [0.3 + 0.6 * i / max(n - 1, 1) for i in range(n)])

PALETTE_MAIN     = '#2A788E'   # viridis teal midpoint
PALETTE_NEG      = '#7E7E7E'
PALETTE_ACCENT   = '#FDE725'   # viridis yellow — secondary lines/areas
PALETTE_LINE_CUM = '#D62728'   # red for cumulative Pareto line

sns.set_theme(style='ticks', font='DejaVu Sans')
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
MLF_RUN_NAME  = '4c530133b755446e9466e3fdee71b5b8'


def save_fig(fig, name: str):
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    if os.path.exists(path):
        os.remove(path)
    if hasattr(fig, 'write_image'):  # plotly
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.write_image(path, width=1400, height=fig.layout.height or 600, scale=2)
        try:
            mlflow.log_artifact(path, artifact_path='figures')
        except Exception:
            pass
        try:
            displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception:
            fig.show()
    else:  # matplotlib
        fig.savefig(path, bbox_inches='tight', facecolor='white')
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

fig = make_subplots(specs=[[{'secondary_y': True}]])

fig.add_trace(go.Bar(
    x=anos, y=df_anual['total_apolices'],
    name='Apólices',
    marker_color=PALETTE_MAIN,
    text=[f'{v:,.0f}' for v in df_anual['total_apolices']],
    textposition='outside',
    textfont=dict(size=10, color='#1E293B'),
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=anos, y=df_anual['taxa_sinistro'],
    name='Taxa de Sinistro',
    mode='lines+markers+text',
    line=dict(color=PALETTE_ACCENT, dash='dash', width=2.5),
    marker=dict(size=7, color=PALETTE_ACCENT, line=dict(color='#333', width=1)),
    text=[f'{v:.1%}' for v in df_anual['taxa_sinistro']],
    textposition='top center',
    textfont=dict(size=9, color='#997700'),
), secondary_y=True)

fig.update_layout(
    title='Evolução do Volume de Apólices e Taxa de Sinistro (2016–2025)',
    template=PLOTLY_TEMPLATE,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    yaxis=dict(title='Total de Apólices', tickformat=',.0f', rangemode='tozero'),
    yaxis2=dict(title='Taxa de Sinistro', tickformat='.1%', rangemode='tozero'),
    xaxis=dict(title='Ano'),
    bargap=0.3,
    height=500,
)
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

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Total de Apólices por Região', 'Taxa de Sinistro por Região'],
    specs=[[{'secondary_y': True}, {'secondary_y': False}]],
    horizontal_spacing=0.12,
)

cum_reg = (df_regiao['total'].cumsum() / df_regiao['total'].sum() * 100).values

fig.add_trace(go.Bar(
    x=df_regiao['regiao'], y=df_regiao['total'],
    name='Total de Apólices',
    marker_color=_viridis(n),
    text=[f'{v:,.0f}' for v in df_regiao['total']],
    textposition='outside',
    textfont=dict(size=9),
    showlegend=False,
), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=df_regiao['regiao'], y=cum_reg,
    name='% Acumulado',
    mode='lines+markers+text',
    line=dict(color=PALETTE_LINE_CUM, dash='dash', width=2),
    marker=dict(size=6, color=PALETTE_LINE_CUM),
    text=[f'{v:.0f}%' for v in cum_reg],
    textposition='top center',
    textfont=dict(size=9, color=PALETTE_LINE_CUM),
    showlegend=False,
), row=1, col=1, secondary_y=True)

fig.add_trace(go.Bar(
    x=df_regiao_taxa['taxa'] * 100, y=df_regiao_taxa['regiao'],
    orientation='h',
    name='Taxa de Sinistro',
    marker_color=_greens(n),
    text=[f'{v:.1f}%' for v in df_regiao_taxa['taxa'] * 100],
    textposition='outside',
    textfont=dict(size=9),
    showlegend=False,
), row=1, col=2)

fig.update_layout(
    title='Distribuição Geográfica do PSR (2016–2025)',
    template=PLOTLY_TEMPLATE,
    height=500,
)
fig.update_yaxes(title_text='Total de Apólices', tickformat=',.0f', row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text='% Acumulado', ticksuffix='%', range=[0, 115], row=1, col=1, secondary_y=True)
fig.update_xaxes(title_text='Taxa de Sinistro (%)', ticksuffix='%', tickformat='.1f', row=1, col=2)

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

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Total de Apólices por Tipo de Cultura', 'Taxa de Sinistro por Tipo de Cultura'],
    specs=[[{'secondary_y': True}, {'secondary_y': False}]],
    horizontal_spacing=0.14,
)

cum_cult = (df_cultura['total'].cumsum() / df_cultura['total'].sum() * 100).values

fig.add_trace(go.Bar(
    x=df_cultura['cultura_fmt'], y=df_cultura['total'],
    name='Total de Apólices',
    marker_color=_viridis(n_c),
    text=[f'{v:,.0f}' for v in df_cultura['total']],
    textposition='outside',
    textfont=dict(size=8),
    showlegend=False,
), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=df_cultura['cultura_fmt'], y=cum_cult,
    name='% Acumulado',
    mode='lines+markers+text',
    line=dict(color=PALETTE_LINE_CUM, dash='dash', width=2),
    marker=dict(size=5, color=PALETTE_LINE_CUM),
    text=[f'{v:.0f}%' for v in cum_cult],
    textposition='top center',
    textfont=dict(size=7.5, color=PALETTE_LINE_CUM),
    showlegend=False,
), row=1, col=1, secondary_y=True)

fig.add_trace(go.Bar(
    x=df_cultura_taxa['taxa'] * 100, y=df_cultura_taxa['cultura_fmt'],
    orientation='h',
    name='Taxa de Sinistro',
    marker_color=_greens(n_c),
    text=[f'{v:.1f}%' for v in df_cultura_taxa['taxa'] * 100],
    textposition='outside',
    textfont=dict(size=8),
    showlegend=False,
), row=1, col=2)

fig.add_shape(
    type='line',
    x0=taxa_media_global * 100, x1=taxa_media_global * 100,
    y0=0, y1=1, yref='paper',
    line=dict(color=PALETTE_LINE_CUM, dash='dash', width=1.5),
    row=1, col=2,
)
fig.add_annotation(
    x=taxa_media_global * 100, y=0.97, yref='paper',
    text=f'Média: {taxa_media_global:.1%}',
    showarrow=False,
    font=dict(size=9, color=PALETTE_LINE_CUM),
    xanchor='left',
    row=1, col=2,
)

fig.update_layout(
    title='Concentração por Tipo de Cultura — PSR (2016–2025)',
    template=PLOTLY_TEMPLATE,
    height=560,
)
fig.update_yaxes(title_text='Total de Apólices', tickformat=',.0f', row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text='% Acumulado', ticksuffix='%', range=[0, 115], row=1, col=1, secondary_y=True)
fig.update_xaxes(title_text='Taxa de Sinistro (%)', ticksuffix='%', tickformat='.1f', row=1, col=2)
fig.update_xaxes(tickangle=-30, row=1, col=1)

save_fig(fig, 'fig_1_3_cultura')

# COMMAND ----------

# DBTITLE 1,[4.1.4] Distribuição da variável resposta
sinistro_counts = df_silver['sinistro'].value_counts().sort_index()
taxa_sinistro   = df_silver['sinistro'].mean()
total           = len(df_silver)

labels_donut = [
    f'Sem Sinistro<br>{sinistro_counts[0]:,.0f} ({sinistro_counts[0]/total:.1%})',
    f'Com Sinistro<br>{sinistro_counts[1]:,.0f} ({sinistro_counts[1]/total:.1%})',
]

fig = go.Figure(go.Pie(
    labels=labels_donut,
    values=sinistro_counts.values.tolist(),
    hole=0.5,
    marker_colors=[PALETTE_NEG, PALETTE_MAIN],
    textinfo='label',
    textfont_size=12,
    hovertemplate='%{label}<extra></extra>',
))
fig.add_annotation(
    text=f'Taxa<br><b>{taxa_sinistro:.1%}</b>',
    x=0.5, y=0.5,
    font_size=16, showarrow=False,
    xanchor='center', yanchor='middle',
)
fig.update_layout(
    title='Distribuição da Variável Resposta (flSinistro)',
    template=PLOTLY_TEMPLATE,
    height=500,
)
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

colors_box = {0: PALETTE_NEG, 1: PALETTE_MAIN}
names_box   = {0: 'Sem sinistro', 1: 'Com sinistro'}

fig = make_subplots(rows=2, cols=3, subplot_titles=VARS_COMPARATIVO)
shown = {0: False, 1: False}

for idx, var in enumerate(VARS_COMPARATIVO):
    r, c = idx // 3 + 1, idx % 3 + 1
    for lv in [0, 1]:
        subset = df_silver[df_silver['sinistro'] == lv][var].dropna()
        q1, med, q3 = subset.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo  = max(float(subset.min()), q1 - 1.5 * iqr)
        hi  = min(float(subset.max()), q3 + 1.5 * iqr)
        fig.add_trace(go.Box(
            q1=[q1], median=[med], q3=[q3],
            lowerfence=[lo], upperfence=[hi],
            mean=[float(subset.mean())],
            x=[names_box[lv]],
            name=names_box[lv],
            marker_color=colors_box[lv],
            showlegend=(not shown[lv]),
            legendgroup=names_box[lv],
        ), row=r, col=c)
        shown[lv] = True

fig.update_layout(
    title='Distribuição das Variáveis Financeiras por Ocorrência de Sinistro',
    template=PLOTLY_TEMPLATE,
    height=600,
    legend=dict(orientation='h', y=1.04),
    boxmode='group',
)
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

fig = make_subplots(specs=[[{'secondary_y': True}]])

fig.add_trace(go.Bar(
    x=df_eventos['evento_fmt'], y=df_eventos['pct'] * 100,
    name='Percentual',
    marker_color=_viridis(n_evt),
    text=[f'{v:.1f}%' for v in df_eventos['pct'] * 100],
    textposition='outside',
    textfont=dict(size=9),
    showlegend=False,
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=df_eventos['evento_fmt'], y=cum_evt,
    name='% Acumulado',
    mode='lines+markers+text',
    line=dict(color=PALETTE_LINE_CUM, dash='dash', width=2),
    marker=dict(size=6, color=PALETTE_LINE_CUM),
    text=[f'{v:.0f}%' for v in cum_evt],
    textposition='top center',
    textfont=dict(size=8, color=PALETTE_LINE_CUM),
    showlegend=False,
), secondary_y=True)

fig.update_layout(
    title='Eventos Causadores de Sinistro — PSR (2016–2025)',
    template=PLOTLY_TEMPLATE,
    height=500,
    xaxis=dict(title='Evento', tickangle=-30),
    yaxis=dict(title='Percentual (%)', ticksuffix='%', tickformat='.1f'),
    yaxis2=dict(title='% Acumulado', ticksuffix='%', range=[0, 115]),
)
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

z_vals   = pivot_test.values.tolist()
x_labels = pivot_test.columns.tolist()
y_labels = pivot_test.index.tolist()

fig = go.Figure(go.Heatmap(
    z=z_vals,
    x=x_labels,
    y=y_labels,
    colorscale='Viridis',
    text=[[f'{v:.3f}' for v in row] for row in z_vals],
    texttemplate='%{text}',
    textfont=dict(size=12),
    showscale=True,
    colorbar=dict(title='Valor'),
))
fig.update_layout(
    title='Desempenho dos Modelos Baseline — Conjunto de Teste',
    template=PLOTLY_TEMPLATE,
    height=360,
    yaxis=dict(autorange='reversed'),
)
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
fig = go.Figure()
fig.add_trace(go.Bar(
    x=labels_plot, y=baseline_vals,
    name='Baseline',
    marker_color=vc[0],
    text=[f'{v:.3f}' if not np.isnan(v) else 'N/D' for v in baseline_vals],
    textposition='outside', textfont=dict(size=10),
))
fig.add_trace(go.Bar(
    x=labels_plot, y=tuned_vals,
    name='Ajustado (GridSearchCV)',
    marker_color=vc[1],
    text=[f'{v:.3f}' if not np.isnan(v) else 'N/D' for v in tuned_vals],
    textposition='outside', textfont=dict(size=10),
))
fig.update_layout(
    title=f'Comparativo Baseline vs. Modelo Ajustado — {MODEL_LABELS[champion]}',
    template=PLOTLY_TEMPLATE,
    barmode='group',
    legend=dict(orientation='h', y=1.05),
    yaxis=dict(title='Valor', tickformat='.3f'),
    height=480,
)
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

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr_test, y=tpr_test,
    mode='lines',
    name=f'Teste (AUC = {auc_test:.3f})',
    line=dict(color=PALETTE_MAIN, width=2.5),
))
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Aleatório',
    line=dict(color='#94A3B8', width=1, dash='dot'),
))
fig.update_layout(
    title='Curva ROC — Conjunto de Teste',
    template=PLOTLY_TEMPLATE,
    xaxis=dict(title='Taxa de Falsos Positivos', tickformat='.1%'),
    yaxis=dict(title='Taxa de Verdadeiros Positivos (Sensibilidade)', tickformat='.1%'),
    legend=dict(x=0.6, y=0.1),
    height=520,
)
save_fig(fig, 'fig_3_1_roc_curves')

# COMMAND ----------

# DBTITLE 1,[4.3.2] Curva Precision-Recall — Conjunto de Teste
prec_test, rec_test, _ = precision_recall_curve(y_test, y_prob_test)
ap_test = average_precision_score(y_test, y_prob_test)
taxa_sinistro_global = df_silver['sinistro'].mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rec_test, y=prec_test,
    mode='lines',
    name=f'Teste (AP = {ap_test:.3f})',
    line=dict(color=PALETTE_MAIN, width=2.5),
))
fig.add_hline(
    y=taxa_sinistro_global,
    line_dash='dot', line_color='#94A3B8', line_width=1.5,
    annotation_text=f'Baseline aleatório: {taxa_sinistro_global:.1%}',
    annotation_position='top right',
)
fig.update_layout(
    title='Curva Precision-Recall — Conjunto de Teste',
    template=PLOTLY_TEMPLATE,
    xaxis=dict(title='Recall', tickformat='.1%'),
    yaxis=dict(title='Precisão', tickformat='.1%'),
    legend=dict(x=0.55, y=0.9),
    height=520,
)
save_fig(fig, 'fig_3_2_pr_curves')

# COMMAND ----------

# DBTITLE 1,[4.3.3] Distribuição de probabilidades preditas por classe
scores_test  = y_prob_test
labels_test  = y_test.values
ks_stat_test = ks_2samp(scores_test[labels_test == 1], scores_test[labels_test == 0]).statistic
scores_oot   = df_oot[SCORE_COL].values

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f'Conjunto de Teste (KS = {ks_stat_test:.3f})',
        f'Período OOT — Monitoramento (n={len(scores_oot):,})',
    ],
)

fig.add_trace(go.Histogram(
    x=scores_test[labels_test == 0], name='Sem sinistro',
    marker_color=PALETTE_NEG, opacity=0.65,
    nbinsx=50, histnorm='probability density',
), row=1, col=1)
fig.add_trace(go.Histogram(
    x=scores_test[labels_test == 1], name='Com sinistro',
    marker_color=PALETTE_MAIN, opacity=0.65,
    nbinsx=50, histnorm='probability density',
), row=1, col=1)

fig.add_trace(go.Histogram(
    x=scores_oot, name='OOT',
    marker_color=PALETTE_MAIN, opacity=0.8,
    nbinsx=50, histnorm='probability density',
), row=1, col=2)
fig.add_shape(
    type='line',
    x0=scores_oot.mean(), x1=scores_oot.mean(),
    y0=0, y1=1, yref='paper',
    line=dict(color=PALETTE_LINE_CUM, dash='dash', width=1.5),
    row=1, col=2,
)
fig.add_annotation(
    x=scores_oot.mean(), y=0.95, yref='paper',
    text=f'Média: {scores_oot.mean():.3f}',
    showarrow=False,
    font=dict(size=9, color=PALETTE_LINE_CUM),
    xanchor='left',
    row=1, col=2,
)

fig.update_layout(
    title='Distribuição dos Escores Preditos',
    template=PLOTLY_TEMPLATE,
    barmode='overlay',
    legend=dict(orientation='h', y=1.05),
    height=480,
)
fig.update_xaxes(title_text='Probabilidade Predita de Sinistro', tickformat='.2f')
fig.update_yaxes(title_text='Densidade')
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

# Top 15; cumulative % computed against ALL features
df_top15_fi = df_fi.head(15).copy()
df_top15_fi['feature_fmt'] = df_top15_fi['feature'].map(_fmt_fi_name)
cum_fi     = (df_top15_fi['importance_pct'].cumsum() * 100).values
bar_colors = df_top15_fi['grupo'].map(GROUP_COLORS).fillna('#CBD5E1').tolist()

fig = make_subplots(specs=[[{'secondary_y': True}]])

fig.add_trace(go.Bar(
    x=df_top15_fi['feature_fmt'], y=df_top15_fi['importance_pct'] * 100,
    name='Importância',
    marker_color=bar_colors,
    text=[f'{v:.1f}%' for v in df_top15_fi['importance_pct'] * 100],
    textposition='outside',
    textfont=dict(size=9),
    showlegend=False,
), secondary_y=False)

# Cumulative line in PALETTE_ACCENT (yellow) — distinct from bars
fig.add_trace(go.Scatter(
    x=df_top15_fi['feature_fmt'], y=cum_fi,
    name='% Acumulado (todas)',
    mode='lines+markers+text',
    line=dict(color=PALETTE_ACCENT, dash='dash', width=2.5),
    marker=dict(size=7, color=PALETTE_ACCENT, line=dict(color='#888', width=1)),
    text=[f'{v:.0f}%' for v in cum_fi],
    textposition='top center',
    textfont=dict(size=8, color='#888'),
), secondary_y=True)

# Invisible traces just to populate the group legend
shown_groups = set(df_top15_fi['grupo'].unique())
for grp, col in GROUP_COLORS.items():
    if grp in shown_groups:
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name=grp,
            marker_color=col,
            showlegend=True,
        ), secondary_y=False)

fig.update_layout(
    title='Top-15 Variáveis por Importância — Modelo Campeão',
    template=PLOTLY_TEMPLATE,
    height=580,
    legend=dict(title='Grupo', orientation='v', x=1.08, y=0.95),
    yaxis=dict(title='Importância Relativa (%)', ticksuffix='%', tickformat='.1f'),
    yaxis2=dict(title='% Acumulado (todas variáveis)', ticksuffix='%', range=[0, 115]),
    xaxis=dict(tickangle=-35),
    bargap=0.25,
)
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

fig = go.Figure(go.Bar(
    x=df_group_imp['Importância Agregada (%)'],
    y=df_group_imp['Grupo'],
    orientation='h',
    marker_color=[GROUP_COLORS.get(g, '#CBD5E1') for g in df_group_imp['Grupo']],
    text=[f"{v:.1f}%" for v in df_group_imp['Importância Agregada (%)']],
    textposition='outside',
    textfont=dict(size=10),
))
fig.update_layout(
    title='Importância por Grupo de Features',
    template=PLOTLY_TEMPLATE,
    height=450,
    xaxis=dict(title='Importância Agregada (%)', ticksuffix='%', tickformat='.1f'),
    yaxis=dict(autorange='reversed'),
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
