-- =============================================================================
-- fs_historico_municipio.sql
-- Feature Store — Histórico de sinistros e severidade por município
--
-- Chave de entidade : mun  (código IBGE do município)
-- Referência temporal: DATE_TRUNC('MONTH', dt_inicio_vigencia)
--
-- dtRef é derivado diretamente de dt_inicio_vigencia (primeiro dia do mês).
-- Como os dados referem-se ao passado, não há parâmetro externo de data.
--
-- Semântica point-in-time preservada via self-join:
--   para cada (dtRef, mun), as features são calculadas sobre apólices cujo
--   dt_fim_vigencia cai ANTES de dtRef — nunca sobre o próprio período.
--   Isso evita que evento/indenizacao do mês corrente contaminem as features.
-- =============================================================================

WITH base AS (
    SELECT
        mun,
        apolice,
        DATE_TRUNC('MONTH', dt_inicio_vigencia) AS dtRefMes,
        dt_fim_vigencia,
        evento,
        total_seg,
        indenizacao
    FROM 02_silver.seg_rural.seg_cleaned
),
meses_mun AS (
    -- Produto cartesiano de meses de referência × municípios presentes nos dados
    SELECT DISTINCT dtRefMes, mun FROM base
)

SELECT
    mm.dtRefMes AS dtRef,
    mm.mun,

    -- ── Janela 365 dias ──────────────────────────────────────────────────────
    COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 365)
               AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.apolice END)
        AS nrApolicesMun365d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 365)
             AND h.dt_fim_vigencia <  mm.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        AS nrSinistrosMun365d,

    -- Taxa de sinistro (% de apólices acionadas no município)
    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 365)
             AND h.dt_fim_vigencia <  mm.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
    / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 365)
                        AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.apolice END), 0)
        AS nrTaxaSinistroMun365d,

    -- Índice de severidade: indenização paga / valor total segurado
    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 365)
             AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.indenizacao ELSE 0 END)
    / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 365)
                      AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.total_seg ELSE 0 END), 0)
        AS nrIndiceSeveridadeMun365d,

    -- ── Janela 730 dias (2 anos) ─────────────────────────────────────────────
    COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 730)
               AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.apolice END)
        AS nrApolicesMun730d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 730)
             AND h.dt_fim_vigencia <  mm.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        AS nrSinistrosMun730d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 730)
             AND h.dt_fim_vigencia <  mm.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
    / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 730)
                        AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.apolice END), 0)
        AS nrTaxaSinistroMun730d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 730)
             AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.indenizacao ELSE 0 END)
    / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 730)
                      AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.total_seg ELSE 0 END), 0)
        AS nrIndiceSeveridadeMun730d

        -- ── Janela 1095 dias (3 anos) ─────────────────────────────────────────────
    COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 1095)
               AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.apolice END)
        AS nrApolicesMun1095d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 1095)
             AND h.dt_fim_vigencia <  mm.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        AS nrSinistrosMun1095d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 1095)
             AND h.dt_fim_vigencia <  mm.dtRefMes
             AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
    / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 1095)
                        AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.apolice END), 0)
        AS nrTaxaSinistroMun1095d,

    SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 1095)
             AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.indenizacao ELSE 0 END)
    / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mm.dtRefMes, 1095)
                      AND h.dt_fim_vigencia <  mm.dtRefMes THEN h.total_seg ELSE 0 END), 0)
        AS nrIndiceSeveridadeMun1095d

FROM meses_mun mm
LEFT JOIN base h ON h.mun = mm.mun
GROUP BY mm.dtRefMes, mm.mun
