-- =============================================================================
-- fl_sinistro.sql
-- Definição da variável resposta (label) para o modelo de sinistro
--
-- A data de referência é o PRIMEIRO DIA DO MÊS de dt_inicio_vigencia.
-- Convenção: dtRef = DATE_TRUNC('MONTH', dt_inicio_vigencia)
-- Isso alinha o join com as Feature Store tables (fs_apolice_financeiro,
-- fs_historico_municipio, fs_risco_cultura_uf), que também usam granularidade
-- mensal — evitando features nulas silenciosas no FeatureLookup.
--
-- Apenas apólices com ciclo completo (dt_fim_vigencia < current_date) são
-- incluídas para garantir que o desfecho seja definitivo no treino.
--
-- Mapeamento de colunas:
--   Chaves primárias  : apolice, dtRef
--   Chaves de lookup  : mun, uf, cultura, tipo_cultura, seguradora
--                       (passadas como exclude_columns em fe.create_training_set)
--   Label             : flSinistro
-- =============================================================================

SELECT
    apolice,
    DATE_TRUNC('MONTH', dt_inicio_vigencia)             AS dtRef,

    -- ── Chaves de join para FeatureLookup ────────────────────────────────────
    -- Necessárias para resolver os joins nas 6 feature tables:
    --   fs_historico_municipio      → mun
    --   fs_risco_cultura_uf         → uf, tipo_cultura
    --   fs_risco_seguradora_cultura → seguradora, tipo_cultura
    --   fs_anomalia_taxa            → cultura, uf
    --   fs_concentracao_carteira    → seguradora, mun
    mun,
    uf,
    cultura,
    tipo_cultura,
    seguradora,

    -- ── Label (variável resposta — classificação binária) ────────────────────
    CASE
        WHEN evento != 'nenhum' THEN 1
        ELSE 0
    END                                                 AS flSinistro

    -- dsEvento, vlIndenizacaoRealizada e nrSinistralidade removidos da âncora:
    -- são labels auxiliares que pertencem à tabela de labels Gold,
    -- não ao training set de classificação binária.

FROM 02_silver.seg_rural.seg_cleaned

-- Somente apólices cujo ciclo já encerrou (desfecho conhecido)
WHERE dt_fim_vigencia < current_date()
