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
--   para cada (dtRef, uf, tipo_cultura), conta apólices cujo dt_fim_vigencia cai
--   ANTES de dtRef — nunca as do próprio período (evita leakage de labels).
--
-- Propósito: fornece uma linha de base de risco para municípios com poucos
-- dados históricos (fallback via hierarquia estado → município).
-- =============================================================================

WITH base AS (
    SELECT
        uf,
        tipo_cultura,
        seguradora,
        DATE_TRUNC('MONTH', dt_inicio_vigencia) AS dtRefMes,
        dt_fim_vigencia,
        evento,
        total_seg,
        indenizacao,
        nivel_cob
    FROM 02_silver.seg_rural.seg_cleaned
),
meses_uf_cultura AS (
    -- Produto cartesiano de meses de referência × (uf, tipo_cultura) presentes nos dados
    SELECT DISTINCT dtRefMes, uf, tipo_cultura FROM base
),
-- Participação de mercado por seguradora em janela de 365 dias — base para HHI
hhi_prep AS (
    SELECT
        mc.dtRefMes,
        mc.uf,
        mc.tipo_cultura,
        h.seguradora,
        COUNT(*)                                                                    AS cnt_seg,
        SUM(COUNT(*)) OVER (PARTITION BY mc.dtRefMes, mc.uf, mc.tipo_cultura)      AS total_cnt
    FROM meses_uf_cultura mc
    LEFT JOIN base h
           ON  h.uf           = mc.uf
           AND h.tipo_cultura = mc.tipo_cultura
           AND h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
           AND h.dt_fim_vigencia <  mc.dtRefMes
    GROUP BY mc.dtRefMes, mc.uf, mc.tipo_cultura, h.seguradora
),
-- Índice Herfindahl-Hirschman (HHI): SUM(share²) por (dtRef, uf, tipo_cultura)
hhi_calc AS (
    SELECT
        dtRefMes,
        uf,
        tipo_cultura,
        SUM(POWER(cnt_seg * 1.0 / NULLIF(total_cnt, 0), 2)) AS nrConcentracaoSeguradora365d
    FROM hhi_prep
    GROUP BY dtRefMes, uf, tipo_cultura
)

SELECT
    mc.dtRefMes             AS dtRef,
    mc.uf,
    mc.tipo_cultura,

    -- ── Janela 365 dias ──────────────────────────────────────────────────────
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
        AS nrSeveridadeCulturaUf365d,

    -- Nível de cobertura médio contratado para a cultura no estado (365d)
    -- Proxy do apetite de risco percebido pelo mercado
    AVG(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
             AND h.dt_fim_vigencia <  mc.dtRefMes THEN h.nivel_cob END)
        AS nrNivelCobMedioCulturaUf365d,

    -- ── Janela 730 dias (2 anos) ─────────────────────────────────────────────
    -- Culturas perenes (café, cana-de-açúcar, pecuário) têm ciclos > 1 ano;
    -- a janela longa estabiliza a taxa de sinistro dessas culturas.
    COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
               AND h.dt_fim_vigencia <  mc.dtRefMes THEN 1 END)
        AS nrApolicesCulturaUf730d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
             AND h.dt_fim_vigencia <  mc.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        AS nrSinistrosCulturaUf730d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
             AND h.dt_fim_vigencia <  mc.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
    / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
                        AND h.dt_fim_vigencia <  mc.dtRefMes THEN 1 END), 0)
        AS nrTaxaSinistroCulturaUf730d,

    -- Severidade 730d: razão indenização / valor segurado
    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
             AND h.dt_fim_vigencia <  mc.dtRefMes THEN h.indenizacao ELSE 0 END)
    / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
                      AND h.dt_fim_vigencia <  mc.dtRefMes THEN h.total_seg ELSE 0 END), 0)
        AS nrSeveridadeCulturaUf730d,

    -- ── Concentração de mercado ───────────────────────────────────────────────
    -- HHI de participação de seguradoras (365d): próximo de 1 indica monopólio
    hhi.nrConcentracaoSeguradora365d

FROM meses_uf_cultura mc
LEFT JOIN base h     ON  h.uf           = mc.uf
                     AND h.tipo_cultura = mc.tipo_cultura
LEFT JOIN hhi_calc hhi ON  hhi.dtRefMes     = mc.dtRefMes
                       AND hhi.uf           = mc.uf
                       AND hhi.tipo_cultura = mc.tipo_cultura
GROUP BY mc.dtRefMes, mc.uf, mc.tipo_cultura, hhi.nrConcentracaoSeguradora365d
