-- =============================================================================
-- fs_concentracao_carteira.sql
-- Feature Store — Concentração de carteira por seguradora e município
--
-- Chave de entidade : (seguradora, mun)
-- Referência temporal: DATE_TRUNC('MONTH', dt_inicio_vigencia)
--
-- dtRef é derivado diretamente de dt_inicio_vigencia (primeiro dia do mês).
--
-- Semântica point-in-time preservada via self-join:
--   para cada (dtRef, seguradora, mun), agrega apólices cujo dt_fim_vigencia
--   cai DENTRO da janela de 365 dias ANTES de dtRef — evita leakage de labels.
--
-- Features:
--   nrPctCarteiraSegMun365d  — participação do município no total_seg da
--     seguradora: total_seg(seguradora, mun) / total_seg(seguradora).
--     Alta concentração em um único município expõe a seguradora a evento
--     climático localizado.
--
--   nrHHI_seguradora_mun — Índice Herfindahl-Hirschman da carteira da
--     seguradora no espaço de municípios: SUM(share_mun²).
--     HHI próximo de 1 indica carteira concentrada em poucos municípios;
--     próximo de 0 indica carteira diversificada.
-- =============================================================================

WITH base AS (
    SELECT
        seguradora,
        mun,
        DATE_TRUNC('MONTH', dt_inicio_vigencia) AS dtRefMes,
        dt_fim_vigencia,
        total_seg
    FROM 02_silver.seg_rural.seg_cleaned
),
chaves AS (
    -- Produto cartesiano de meses de referência × (seguradora, mun)
    SELECT DISTINCT dtRefMes, seguradora, mun FROM base
),
-- Agregação por (dtRef, seguradora, mun) — capital segurado do par no período
agg_seg_mun AS (
    SELECT
        c.dtRefMes,
        c.seguradora,
        c.mun,
        SUM(CASE
            WHEN h.dt_fim_vigencia >= DATE_SUB(c.dtRefMes, 365)
             AND h.dt_fim_vigencia <  c.dtRefMes
            THEN h.total_seg ELSE 0
        END)                                                            AS ts_seg_mun
    FROM chaves c
    LEFT JOIN base h
           ON  h.seguradora = c.seguradora
           AND h.mun        = c.mun
    GROUP BY c.dtRefMes, c.seguradora, c.mun
),
-- Total do capital segurado da seguradora no período (todos os municípios)
total_seg_segurador AS (
    SELECT
        dtRefMes,
        seguradora,
        SUM(ts_seg_mun)                                                 AS ts_total_seg
    FROM agg_seg_mun
    GROUP BY dtRefMes, seguradora
),
-- Share do município no capital segurado da seguradora
shares AS (
    SELECT
        sm.dtRefMes,
        sm.seguradora,
        sm.mun,
        sm.ts_seg_mun,
        sm.ts_seg_mun / NULLIF(t.ts_total_seg, 0)                      AS share_mun
    FROM agg_seg_mun sm
    JOIN total_seg_segurador t
      ON  t.dtRefMes   = sm.dtRefMes
      AND t.seguradora = sm.seguradora
),
-- HHI da seguradora: SUM(share_mun²) — calculado sobre todos os municípios
hhi_calc AS (
    SELECT
        dtRefMes,
        seguradora,
        SUM(POWER(share_mun, 2))                                        AS nrHHI_seguradora_mun
    FROM shares
    GROUP BY dtRefMes, seguradora
)

SELECT
    s.dtRefMes                                                          AS dtRef,
    s.seguradora,
    s.mun,
    s.share_mun                                                         AS nrPctCarteiraSegMun365d,
    h.nrHHI_seguradora_mun
FROM shares s
JOIN hhi_calc h
  ON  h.dtRefMes   = s.dtRefMes
  AND h.seguradora = s.seguradora
