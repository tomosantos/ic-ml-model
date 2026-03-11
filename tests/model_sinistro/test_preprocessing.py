import numpy as np
import pandas as pd
import pytest

from src.model_sinistro.preprocessing import (
    FEATURES_CATEGORICAS,
    FEATURES_CICLICAS,
    FEATURES_NUMERICAS_APOLICE,
    FEATURES_NUMERICAS_HISTORICAS,
    apply_tf_preprocessor,
    build_tf_preprocessor,
    derive_features,
    pipeline_linear,
    pipeline_tree,
)

_ALL_COLS = (
    FEATURES_NUMERICAS_HISTORICAS
    + FEATURES_NUMERICAS_APOLICE
    + FEATURES_CATEGORICAS
    + FEATURES_CICLICAS
)

_CAT_VALUES = {
    'tipo_cultura':           ['graos', 'frutas', 'hortalicas', 'perenes'],
    'regiao':                 ['Sul', 'Sudeste', 'Centro-Oeste', 'Nordeste', 'Norte'],
    'seguradora':             ['alianca', 'allianz', 'mapfre', 'tokio_marine'],
    'nrEventosDominante365d': ['seca', 'granizo', 'chuva', 'nenhum'],
}


@pytest.fixture(scope='module')
def X_train() -> pd.DataFrame:
    """DataFrame de treino com NaN estruturais e colunas categóricas."""
    rng = np.random.default_rng(42)
    n = 60

    df = pd.DataFrame(rng.random((n, len(_ALL_COLS))), columns=_ALL_COLS)

    # NaN estruturais — simula municípios sem histórico no FeatureLookup
    nan_idx = rng.choice(n, 20, replace=False)
    df.loc[nan_idx, FEATURES_NUMERICAS_HISTORICAS] = np.nan

    for col, vals in _CAT_VALUES.items():
        df[col] = rng.choice(vals, n).astype(str)

    # Cíclicas em [-1, 1] — nunca nulas
    df['nrSinMes'] = np.sin(rng.uniform(0, 2 * np.pi, n))
    df['nrCosMes'] = np.cos(rng.uniform(0, 2 * np.pi, n))

    return df


@pytest.fixture(scope='module')
def tf_preprocessor(X_train):
    return build_tf_preprocessor(X_train)


# ── derive_features ───────────────────────────────────────────────────────────

class TestDeriveFeatures:
    def test_anomalia_taxa_nan_when_zero_denominator(self):
        df = pd.DataFrame({
            'nrTaxaApolice':           [0.05, 0.10],
            'nrTaxaMediaCulturaUf365d': [0.0, 0.20],
            'nrMesPlantio':            [3, 9],
        })
        out = derive_features(df)
        assert np.isnan(out.loc[0, 'nrAnomaliaTaxa'])
        assert pytest.approx(out.loc[1, 'nrAnomaliaTaxa']) == 0.10 / 0.20

    def test_cyclical_encoding_range(self):
        df = pd.DataFrame({
            'nrTaxaApolice':           [0.05] * 12,
            'nrTaxaMediaCulturaUf365d': [0.10] * 12,
            'nrMesPlantio':            list(range(1, 13)),
        })
        out = derive_features(df)
        assert out['nrSinMes'].between(-1.0, 1.0).all()
        assert out['nrCosMes'].between(-1.0, 1.0).all()

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            'nrTaxaApolice':           [0.05],
            'nrTaxaMediaCulturaUf365d': [0.10],
            'nrMesPlantio':            [6],
        })
        original_cols = set(df.columns)
        derive_features(df)
        assert set(df.columns) == original_cols


# ── pipeline_tree ─────────────────────────────────────────────────────────────

class TestPipelineTree:
    def test_fit_transform_no_nan(self, X_train):
        out = pipeline_tree.fit_transform(X_train)
        assert not np.isnan(out).any(), "pipeline_tree.fit_transform produziu NaN inesperado"

    def test_transform_unknown_categories(self, X_train):
        X_test = X_train.copy()
        for col in FEATURES_CATEGORICAS:
            X_test[col] = 'CATEGORIA_DESCONHECIDA_EM_PRODUCAO'
        # OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # não deve lançar exceção
        pipeline_tree.transform(X_test)

    def test_output_shape(self, X_train):
        out = pipeline_tree.fit_transform(X_train)
        expected_n_features = (
            len(FEATURES_NUMERICAS_HISTORICAS)
            + len(FEATURES_NUMERICAS_APOLICE)
            + len(FEATURES_CATEGORICAS)
            + len(FEATURES_CICLICAS)
        )
        assert out.shape == (len(X_train), expected_n_features)


# ── pipeline_linear ───────────────────────────────────────────────────────────

class TestPipelineLinear:
    def test_fit_transform_no_nan(self, X_train):
        out = pipeline_linear.fit_transform(X_train)
        assert not np.isnan(out).any()

    def test_same_n_features_as_tree(self, X_train):
        out_tree   = pipeline_tree.fit_transform(X_train)
        out_linear = pipeline_linear.fit_transform(X_train)
        assert out_linear.shape[1] == out_tree.shape[1]


# ── TensorFlow preprocessor ───────────────────────────────────────────────────

class TestTFPreprocessor:
    def test_apply_returns_float32(self, X_train, tf_preprocessor):
        out = apply_tf_preprocessor(tf_preprocessor, X_train)
        assert out.dtype == np.float32

    def test_no_nan_in_output(self, X_train, tf_preprocessor):
        out = apply_tf_preprocessor(tf_preprocessor, X_train)
        assert not np.isnan(out).any()

    def test_same_n_features_as_linear(self, X_train, tf_preprocessor):
        out_tf     = apply_tf_preprocessor(tf_preprocessor, X_train)
        out_linear = pipeline_linear.fit_transform(X_train)
        assert out_tf.shape[1] == out_linear.shape[1]
