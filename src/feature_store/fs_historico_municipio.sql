-- =============================================================================
-- fs_historico_municipio.sql
-- Feature Store — Histórico de sinistros, severidade e densidade por município
--
-- Chave de entidade : mun  (código IBGE do município)
-- dtRef é derivado diretamente de dt_inicio_vigencia (primeiro dia do mês).
--
-- Semântica point-in-time preservada:
--   As features históricas usam apenas apólices com dt_fim_vigencia < dtRef —
--   nunca as do próprio período (evita leakage de labels).
--
-- Features de densidade (nrApolicesAbertas*) usam dt_inicio_vigencia para
-- capturar seleção adversa: alta contratação recente pode indicar evento
-- climático percebido antecipadamente pelos produtores.
--
-- Coluna 'evento' na Silver usa o valor 'nenhum' (não NULL nem '0') para
-- apólices sem sinistro — o predicado evento != 'nenhum' depende disso.
-- =============================================================================

WITH base AS (
    SELECT
        mun,
        apolice,
        dt_inicio_vigencia,
        dt_fim_vigencia,
        evento,
        total_seg,
        indenizacao,
        DATE_TRUNC('MONTH', dt_inicio_vigencia) AS dtRefMes
    FROM 02_silver.seg_rural.seg_cleaned
),
mesh AS (
    -- Todos os pares (dtRef, mun) para os quais calcular features
    SELECT DISTINCT dtRefMes, mun FROM base
),
historico_agg AS (
    -- Para cada (dtRefMes, mun): agrega apólices encerradas ANTES de dtRefMes
    SELECT
        mc.dtRefMes,
        mc.mun,

        -- ── Janela 90 dias ───────────────────────────────────────────────────
        COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 90)
                   THEN h.apolice END)
            AS nrApolicesMun90d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 90)
                 AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
            AS nrSinistrosMun90d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 90)
                 AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 90)
                            THEN h.apolice END), 0)
            AS nrTaxaSinistroMun90d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 90)
                 THEN h.indenizacao ELSE 0 END)
        / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 90)
                          THEN h.total_seg ELSE 0 END), 0)
            AS nrIndiceSeveridadeMun90d,

        -- ── Janela 365 dias ──────────────────────────────────────────────────
        COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
                   THEN h.apolice END)
            AS nrApolicesMun365d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
                 AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
            AS nrSinistrosMun365d,

        -- Taxa de sinistro (% de apólices acionadas no município)
        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
                 AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
                            THEN h.apolice END), 0)
            AS nrTaxaSinistroMun365d,

        -- Índice de severidade: indenização paga / valor total segurado
        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
                 THEN h.indenizacao ELSE 0 END)
        / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
                          THEN h.total_seg ELSE 0 END), 0)
            AS nrIndiceSeveridadeMun365d,

        -- ── Janela 730 dias (2 anos) ─────────────────────────────────────────
        COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
                   THEN h.apolice END)
            AS nrApolicesMun730d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
                 AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
            AS nrSinistrosMun730d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
                 AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
                            THEN h.apolice END), 0)
            AS nrTaxaSinistroMun730d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
                 THEN h.indenizacao ELSE 0 END)
        / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 730)
                          THEN h.total_seg ELSE 0 END), 0)
            AS nrIndiceSeveridadeMun730d,

        -- ── Janela 1095 dias (3 anos) ────────────────────────────────────────
        COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 1095)
                   THEN h.apolice END)
            AS nrApolicesMun1095d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 1095)
                 AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
            AS nrSinistrosMun1095d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 1095)
                 AND h.evento != 'nenhum' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 1095)
                            THEN h.apolice END), 0)
            AS nrTaxaSinistroMun1095d,

        SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 1095)
                 THEN h.indenizacao ELSE 0 END)
        / NULLIF(SUM(CASE WHEN h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 1095)
                          THEN h.total_seg ELSE 0 END), 0)
            AS nrIndiceSeveridadeMun1095d

    FROM mesh mc
    LEFT JOIN base h
           ON h.mun            = mc.mun
          AND h.dt_fim_vigencia < mc.dtRefMes
    GROUP BY mc.dtRefMes, mc.mun
),
aberturas_agg AS (
    -- Apólices abertas recentemente — proxy de seleção adversa
    SELECT
        mc.dtRefMes,
        mc.mun,
        COUNT(CASE WHEN h.dt_inicio_vigencia >= DATE_SUB(mc.dtRefMes, 30)
                   THEN h.apolice END) AS nrApolicesAbertas30d,
        COUNT(CASE WHEN h.dt_inicio_vigencia >= DATE_SUB(mc.dtRefMes, 90)
                   THEN h.apolice END) AS nrApolicesAbertas90d
    FROM mesh mc
    LEFT JOIN base h
           ON h.mun                = mc.mun
          AND h.dt_inicio_vigencia >= DATE_SUB(mc.dtRefMes, 90)
          AND h.dt_inicio_vigencia <  mc.dtRefMes
    GROUP BY mc.dtRefMes, mc.mun
),
evento_dominante AS (
    -- Evento mais frequente por (dtRef, mun) nos últimos 365 dias
    SELECT
        mc.dtRefMes,
        mc.mun,
        h.evento,
        ROW_NUMBER() OVER (
            PARTITION BY mc.dtRefMes, mc.mun
            ORDER BY COUNT(*) DESC
        ) AS rn
    FROM mesh mc
    JOIN base h
      ON h.mun            = mc.mun
     AND h.dt_fim_vigencia >= DATE_SUB(mc.dtRefMes, 365)
     AND h.dt_fim_vigencia <  mc.dtRefMes
     AND h.evento         != 'nenhum'
    GROUP BY mc.dtRefMes, mc.mun, h.evento
)

SELECT
    mc.dtRefMes                                   AS dtRef,
    mc.mun,

    h.nrApolicesMun90d,
    h.nrSinistrosMun90d,
    h.nrTaxaSinistroMun90d,
    h.nrIndiceSeveridadeMun90d,

    h.nrApolicesMun365d,
    h.nrSinistrosMun365d,
    h.nrTaxaSinistroMun365d,
    h.nrIndiceSeveridadeMun365d,

    h.nrApolicesMun730d,
    h.nrSinistrosMun730d,
    h.nrTaxaSinistroMun730d,
    h.nrIndiceSeveridadeMun730d,

    h.nrApolicesMun1095d,
    h.nrSinistrosMun1095d,
    h.nrTaxaSinistroMun1095d,
    h.nrIndiceSeveridadeMun1095d,

    -- ── Seleção adversa: densidade de novas apólices ─────────────────────────
    COALESCE(a.nrApolicesAbertas30d, 0)           AS nrApolicesAbertas30d,
    COALESCE(a.nrApolicesAbertas90d, 0)           AS nrApolicesAbertas90d,

    -- ── Evento dominante nos últimos 365 dias (categórica para encoding) ──────
    ed.evento                                     AS nrEventosDominante365d

FROM mesh mc
LEFT JOIN historico_agg    h  ON h.dtRefMes  = mc.dtRefMes AND h.mun  = mc.mun
LEFT JOIN aberturas_agg    a  ON a.dtRefMes  = mc.dtRefMes AND a.mun  = mc.mun
LEFT JOIN evento_dominante ed ON ed.dtRefMes = mc.dtRefMes AND ed.mun = mc.mun AND ed.rn = 1
