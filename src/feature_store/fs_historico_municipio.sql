-- =============================================================================
-- fs_historico_municipio.sql
-- Feature Store — Histórico de sinistros e severidade por município
--
-- Chave de entidade : mun  (código IBGE do município)
-- Referência temporal: {dt_ref}  — deve ser o PRIMEIRO DIA DO MÊS.
--   Convenção: dt_ref = DATE_TRUNC('MONTH', dt_inicio_vigencia)
--   Normalização garantida em compute_feature_store.py antes de formatar.
--
-- Uso:
--   query = open('fs_historico_municipio.sql').read()
--   df = spark.sql(query.format(dt_ref='2024-01-01'))  # sempre dia 1
-- =============================================================================

WITH tb_historico AS (
    SELECT
        mun,
        apolice,
        dt_fim_vigencia,
        evento,
        total_seg,
        indenizacao
    FROM 02_silver.seg_rural.seg_cleaned
    -- Point-in-Time: só apólices cujo ciclo já encerrou ANTES da data de referência.
    -- O desfecho (sinistro/indenizacao) só é conhecido após o fim da vigência.
    WHERE dt_fim_vigencia < '{dt_ref}'
)

SELECT
    '{dt_ref}'  AS dtRef,
    mun,

    -- ── Janela 365 dias ──────────────────────────────────────────────────────
    COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365) THEN apolice END)
        AS nrApolicesMun365d,

    SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
             AND evento != 'nenhum' THEN 1 ELSE 0 END)
        AS nrSinistrosMun365d,

    -- Taxa de sinistro (% de apólices acionadas no município)
    SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
             AND evento != 'nenhum' THEN 1 ELSE 0 END)
    / NULLIF(COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365) THEN apolice END), 0)
        AS nrTaxaSinistroMun365d,

    -- Índice de severidade: indenização paga / valor total segurado
    SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365) THEN indenizacao ELSE 0 END)
    / NULLIF(SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365) THEN total_seg ELSE 0 END), 0)
        AS nrIndiceSeveridadeMun365d,

    -- ── Janela 730 dias (2 anos) ─────────────────────────────────────────────
    COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730) THEN apolice END)
        AS nrApolicesMun730d,

    SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730)
             AND evento != 'nenhum' THEN 1 ELSE 0 END)
        AS nrSinistrosMun730d,

    SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730)
             AND evento != 'nenhum' THEN 1 ELSE 0 END)
    / NULLIF(COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730) THEN apolice END), 0)
        AS nrTaxaSinistroMun730d,

    SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730) THEN indenizacao ELSE 0 END)
    / NULLIF(SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730) THEN total_seg ELSE 0 END), 0)
        AS nrIndiceSeveridadeMun730d

FROM tb_historico
GROUP BY mun
