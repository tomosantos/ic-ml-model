-- =============================================================================
-- fs_anomalia_taxa.sql
-- Feature Store — Parâmetros de referência de taxa por (cultura, UF)
--
-- Chave de entidade : (cultura, uf)
-- Referência temporal: DATE_TRUNC('MONTH', dt_inicio_vigencia)
--
-- dtRef é derivado diretamente de dt_inicio_vigencia (primeiro dia do mês).
--
-- Semântica point-in-time preservada via self-join:
--   para cada (dtRef, cultura, uf), agrega taxas de apólices cujo
--   dt_fim_vigencia cai DENTRO da janela de 365 dias ANTES de dtRef —
--   evita leakage de dados futuros.
--
-- Propósito: provê os parâmetros de referência (média e desvio padrão da taxa)
-- para detecção de anomalia de precificação. O índice de anomalia final
-- (taxa_apolice / nrTaxaMediaCulturaUf365d) é calculado no join com
-- fs_apolice_financeiro em train.py — este módulo expõe apenas os parâmetros.
--
-- Mercados com alta volatilidade (nrStdTaxaCulturaUf365d elevado) indicam
-- risco sistêmico maior e precificação menos estável.
-- =============================================================================

WITH base AS (
    SELECT
        cultura,
        uf,
        DATE_TRUNC('MONTH', dt_inicio_vigencia) AS dtRefMes,
        dt_fim_vigencia,
        taxa
    FROM 02_silver.seg_rural.seg_cleaned
    WHERE taxa IS NOT NULL
),
chaves AS (
    -- Produto cartesiano de meses de referência × (cultura, uf)
    SELECT DISTINCT dtRefMes, cultura, uf FROM base
)

SELECT
    c.dtRefMes                                                          AS dtRef,
    c.cultura,
    c.uf,

    -- ── Janela 365 dias ──────────────────────────────────────────────────────
    COUNT(CASE
        WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
         AND h.dt_fim_vigencia <  c.dtRefMes
        THEN 1
    END)                                                                AS nrApolicesCulturaUf365d,

    -- Taxa média histórica do par (cultura, uf) — referência de precificação
    AVG(CASE
        WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
         AND h.dt_fim_vigencia <  c.dtRefMes
        THEN h.taxa
    END)                                                                AS nrTaxaMediaCulturaUf365d,

    -- Desvio padrão da taxa — indica volatilidade/risco sistêmico do mercado
    STDDEV(CASE
        WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
         AND h.dt_fim_vigencia <  c.dtRefMes
        THEN h.taxa
    END)                                                                AS nrStdTaxaCulturaUf365d

FROM chaves c
LEFT JOIN base h
       ON  h.cultura = c.cultura
       AND h.uf      = c.uf
GROUP BY c.dtRefMes, c.cultura, c.uf
