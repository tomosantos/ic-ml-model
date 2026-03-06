-- =============================================================================
-- fs_risco_cultura_uf.sql
-- Feature Store — Risco base por cultura e estado
--
-- Chave de entidade : (uf, tipo_cultura)
-- Referência temporal: {dt_ref}  — deve ser o PRIMEIRO DIA DO MÊS.
--   Convenção: dt_ref = DATE_TRUNC('MONTH', dt_inicio_vigencia)
--   Normalização garantida em compute_feature_store.py antes de formatar.
--
-- Propósito: fornece uma linha de base de risco para municípios com poucos
-- dados históricos (fallback via hierarquia estado → município).
-- =============================================================================

WITH tb_historico_cultura AS (
    SELECT
        uf,
        tipo_cultura,
        evento,
        total_seg,
        indenizacao
    FROM 02_silver.seg_rural.seg_cleaned
    -- Point-in-Time: apenas apólices encerradas antes da data de referência
    WHERE dt_fim_vigencia <  '{dt_ref}'
      AND dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
)

SELECT
    '{dt_ref}'  AS dtRef,
    uf,
    tipo_cultura,

    COUNT(*)
        AS nrApolicesCulturaUf365d,

    SUM(CASE WHEN evento != 'nenhum' THEN 1 ELSE 0 END)
        AS nrSinistrosCulturaUf365d,

    -- Taxa de sinistro da cultura no estado
    SUM(CASE WHEN evento != 'nenhum' THEN 1 ELSE 0 END)
    / NULLIF(COUNT(*), 0)
        AS nrTaxaSinistroCulturaUf365d,

    -- Severidade média: indenização / valor segurado
    SUM(indenizacao)
    / NULLIF(SUM(total_seg), 0)
        AS nrSeveridadeCulturaUf365d

FROM tb_historico_cultura
GROUP BY uf, tipo_cultura
