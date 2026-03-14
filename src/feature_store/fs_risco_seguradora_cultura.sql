-- =============================================================================
-- fs_risco_seguradora_cultura.sql
-- Feature Store — Comportamento histórico de sinistros por seguradora e cultura
--
-- Chave de entidade : (seguradora, tipo_cultura)
-- Referência temporal: DATE_TRUNC('MONTH', dt_inicio_vigencia)
--
-- dtRef é derivado diretamente de dt_inicio_vigencia (primeiro dia do mês).
--
-- Semântica point-in-time preservada via self-join:
--   para cada (dtRef, seguradora, tipo_cultura), conta apólices cujo
--   dt_fim_vigencia cai DENTRO da janela de 365 dias ANTES de dtRef —
--   evita leakage de labels.
--
-- Propósito: captura o comportamento histórico de cada seguradora por cultura,
-- permitindo identificar padrões de seleção adversa e pricing diferenciado.
-- =============================================================================

WITH base AS (
    SELECT
        seguradora,
        tipo_cultura,
        DATE_TRUNC('MONTH', dt_inicio_vigencia) AS dtRefMes,
        dt_fim_vigencia,
        evento,
        total_seg,
        indenizacao
    FROM 02_silver.seg_rural.seg_cleaned
),
chaves AS (
    -- Produto cartesiano de meses de referência × (seguradora, tipo_cultura)
    SELECT DISTINCT dtRefMes, seguradora, tipo_cultura FROM base
)

SELECT
    c.dtRefMes                                                          AS dtRef,
    c.seguradora,
    c.tipo_cultura,

    -- ── Janela 365 dias ──────────────────────────────────────────────────────
    COUNT(CASE
        WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
         AND h.dt_fim_vigencia <  c.dtRefMes
        THEN 1
    END)                                                                AS nrApolicesSegCultura365d,

    -- Taxa de sinistro: apólices sinistradas / total de apólices encerradas
    SUM(CASE
        WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
         AND h.dt_fim_vigencia <  c.dtRefMes
         AND h.evento != 'nenhum'
        THEN 1 ELSE 0
    END)
    / NULLIF(COUNT(CASE
        WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
         AND h.dt_fim_vigencia <  c.dtRefMes
        THEN 1
    END), 0)                                                            AS nrTaxaSinistroSegCultura365d,

    -- Severidade: indenização total / capital segurado total
    SUM(CASE
        WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
         AND h.dt_fim_vigencia <  c.dtRefMes
        THEN h.indenizacao ELSE 0
    END)
    / NULLIF(SUM(CASE
        WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
         AND h.dt_fim_vigencia <  c.dtRefMes
        THEN h.total_seg ELSE 0
    END), 0)                                                            AS nrSeveridadeSegCultura365d

FROM chaves c
LEFT JOIN base h
       ON  h.seguradora   = c.seguradora
       AND h.tipo_cultura = c.tipo_cultura
GROUP BY c.dtRefMes, c.seguradora, c.tipo_cultura
