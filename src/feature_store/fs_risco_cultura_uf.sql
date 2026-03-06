-- =============================================================================
-- fs_risco_cultura_uf.sql
-- Feature Store — Risco base por cultura e estado
--
-- Chave de entidade : (uf, tipo_cultura)
-- Referência temporal: DATE_TRUNC('MONTH', dt_inicio_vigencia)
--
-- dtRef é derivado diretamente de dt_inicio_vigencia (primeiro dia do mês).
-- Como os dados referem-se ao passado, não há parâmetro externo de data.
--
-- Semântica point-in-time preservada via self-join:
--   para cada (dtRef, uf, tipo_cultura), conta apólices encerradas nos 365 dias
--   ANTERIORES a dtRef — nunca as do próprio período (evita leakage de labels).
--
-- Propósito: fornece uma linha de base de risco para municípios com poucos
-- dados históricos (fallback via hierarquia estado → município).
-- =============================================================================

WITH base AS (
    SELECT
        uf,
        tipo_cultura,
        DATE_TRUNC('MONTH', dt_inicio_vigencia) AS dtRefMes,
        dt_fim_vigencia,
        evento,
        total_seg,
        indenizacao
    FROM 02_silver.seg_rural.seg_cleaned
),
meses_uf_cultura AS (
    -- Produto cartesiano de meses de referência × (uf, tipo_cultura) presentes nos dados
    SELECT DISTINCT dtRefMes, uf, tipo_cultura FROM base
)

SELECT
    mc.dtRefMes             AS dtRef,
    mc.uf,
    mc.tipo_cultura,

    COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
               AND h.dt_fim_vigencia <  mc.dtRefMes THEN 1 END)
        AS nrApolicesCulturaUf365d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
             AND h.dt_fim_vigencia <  mc.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        AS nrSinistrosCulturaUf365d,

    -- Taxa de sinistro da cultura no estado
    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
             AND h.dt_fim_vigencia <  mc.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
    / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
                        AND h.dt_fim_vigencia <  mc.dtRefMes THEN 1 END), 0)
        AS nrTaxaSinistroCulturaUf365d,

    -- Severidade média: indenização / valor segurado
    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
             AND h.dt_fim_vigencia <  mc.dtRefMes THEN h.indenizacao ELSE 0 END)
    / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
                      AND h.dt_fim_vigencia <  mc.dtRefMes THEN h.total_seg ELSE 0 END), 0)
        AS nrSeveridadeCulturaUf365d

FROM meses_uf_cultura mc
LEFT JOIN base h ON h.uf = mc.uf AND h.tipo_cultura = mc.tipo_cultura
GROUP BY mc.dtRefMes, mc.uf, mc.tipo_cultura
