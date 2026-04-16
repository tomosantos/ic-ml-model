import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Nulos estruturais de FeatureLookup:
#   Municípios / combinações sem histórico terão NaN em TODAS as features históricas.
#   Imputação por mediana — nunca zero (zero = histórico limpo, NaN = ausência de histórico).

FEATURES_NUMERICAS_HISTORICAS = [
    # ── fs_historico_municipio ────────────────────────────────────────────────
    'nrApolicesMun90d',        'nrSinistrosMun90d',
    'nrTaxaSinistroMun90d',    'nrIndiceSeveridadeMun90d',
    'nrApolicesMun365d',       'nrSinistrosMun365d',
    'nrTaxaSinistroMun365d',   'nrIndiceSeveridadeMun365d',
    'nrApolicesMun730d',       'nrSinistrosMun730d',
    'nrTaxaSinistroMun730d',   'nrIndiceSeveridadeMun730d',
    'nrApolicesMun1095d',      'nrSinistrosMun1095d',
    'nrTaxaSinistroMun1095d',  'nrIndiceSeveridadeMun1095d',
    'nrApolicesAbertas30d',    'nrApolicesAbertas90d',
    # ── fs_risco_cultura_uf ───────────────────────────────────────────────────
    'nrApolicesCulturaUf365d', 'nrSinistrosCulturaUf365d',
    'nrTaxaSinistroCulturaUf365d', 'nrSeveridadeCulturaUf365d',
    'nrNivelCobMedioCulturaUf365d',
    'nrApolicesCulturaUf730d', 'nrSinistrosCulturaUf730d',
    'nrTaxaSinistroCulturaUf730d', 'nrSeveridadeCulturaUf730d',
    'nrConcentracaoSeguradora365d',
    # ── fs_risco_seguradora_cultura ───────────────────────────────────────────
    'nrApolicesSegCultura365d', 'nrTaxaSinistroSegCultura365d',
    'nrSeveridadeSegCultura365d',
    # ── fs_anomalia_taxa (parâmetros de referência de precificação) ───────────
    'nrApolicesCulturaExata365d',
    'nrTaxaMediaCulturaUf365d', 'nrStdTaxaCulturaUf365d',
    # ── fs_concentracao_carteira ──────────────────────────────────────────────
    'nrPctCarteiraSegMun365d', 'nrHHI_seguradora_mun',
    # ── feature derivada pós-join (calculada por derive_features) ────────────
    'nrAnomaliaTaxa',
]

# Features da apólice (fs_apolice_financeiro) — imputação por mediana como fallback.
# nrMesPlantio é excluído: substituído por nrSinMes / nrCosMes em FEATURES_CICLICAS
# para preservar a natureza circular do calendário (dezembro → janeiro).
FEATURES_NUMERICAS_APOLICE = [
    'nrTrimestre', 'nrAnoPlantio',
    'nrDuracaoDias', 'nrDuracaoRelativa', 'flSafraVerao',
    'nrDensidadeValorSegHa', 'nrPremioPorHa',
    'nrRazaoCoberturaProd',
    'nrRazaoSubvencaoPremio', 'nrTaxaApolice', 'nrNivelCobertura',
    'nrAreaPorAnimal',
]

# Features categóricas — OrdinalEncoder com unknown_value=-1 para categorias novas em produção.
FEATURES_CATEGORICAS = ['tipo_cultura', 'regiao', 'seguradora', 'nrEventosDominante365d']

# Features cíclicas (sin/cos do mês de plantio) — derivadas por derive_features.
# Nunca contêm nulos; passam direto sem transformação via remainder='passthrough'.
FEATURES_CICLICAS = ['nrSinMes', 'nrCosMes']


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features derivadas após o join com FeatureLookup.

    Deve ser chamada por train.py e predict.py antes de qualquer pipeline sklearn ou TF.

    Transformações:
      nrAnomaliaTaxa — nrTaxaApolice / nrTaxaMediaCulturaUf365d
                       NaN quando denominador <= 0, preservando ausência de histórico.
      nrSinMes       — sin(2π × nrMesPlantio / 12)
      nrCosMes       — cos(2π × nrMesPlantio / 12)
      Ver: https://shrmtmt.medium.com/understand-the-capabilities-of-cyclic-encoding-5b68f831387e

    Args:
        df: DataFrame com pelo menos as colunas
            nrTaxaApolice, nrTaxaMediaCulturaUf365d, nrMesPlantio.

    Returns:
        Novo DataFrame com as colunas derivadas adicionadas.
    """
    df = df.copy()
    df['nrAnomaliaTaxa'] = np.where(
        df['nrTaxaMediaCulturaUf365d'] > 0,
        df['nrTaxaApolice'] / df['nrTaxaMediaCulturaUf365d'],
        np.nan,
    )
    df['nrSinMes'] = np.sin(2 * np.pi * df['nrMesPlantio'] / 12)
    df['nrCosMes'] = np.cos(2 * np.pi * df['nrMesPlantio'] / 12)
    return df


# Árvores são invariantes a escala — não há StandardScaler.
# FEATURES_CICLICAS passam via remainder='passthrough' sem qualquer transformação.
pipeline_tree = Pipeline([
    ('col_transform', ColumnTransformer(
        [
            ('num', SimpleImputer(strategy='median'),
             FEATURES_NUMERICAS_HISTORICAS + FEATURES_NUMERICAS_APOLICE),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('enc', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )),
            ]), FEATURES_CATEGORICAS),
        ],
        remainder='passthrough',   # FEATURES_CICLICAS passam direto
    )),
])


# Idêntico ao pipeline_tree com StandardScaler adicional — obrigatório para modelos lineares.
pipeline_linear = Pipeline([
    ('col_transform', ColumnTransformer(
        [
            ('num', SimpleImputer(strategy='median'),
             FEATURES_NUMERICAS_HISTORICAS + FEATURES_NUMERICAS_APOLICE),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('enc', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )),
            ]), FEATURES_CATEGORICAS),
        ],
        remainder='passthrough',   # FEATURES_CICLICAS passam direto
    )),
    ('scaler', StandardScaler()),
])



