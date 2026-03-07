-- =============================================================================
-- fs_historico_municipio.sql
-- Feature Store — Histórico de sinistros, severidade e densidade por município
--
-- Chave de entidade : mun  (código IBGE do município)
-- Referência temporal: '{dt_ref}' (parâmetro externo — primeiro dia de um mês)
--
-- Semântica point-in-time preservada via predicado na CTE base:
--   WHERE dt_fim_vigencia < '{dt_ref}' — garante que apenas apólices encerradas
--   antes de dt_ref entram nas features históricas, evitando leakage de labels.
--
-- Features de densidade (nrApolicesAbertas*) usam dt_inicio_vigencia para
-- capturar seleção adversa: alta contratação recente pode indicar evento
-- climático percebido antecipadamente pelos produtores. São calculadas sobre
-- todas as apólices (abertas ou encerradas) via CTE separada.
--
-- Coluna 'evento' na Silver usa o valor 'nenhum' (não NULL nem '0') para
-- apólices sem sinistro — o predicado evento != 'nenhum' depende disso.
-- =============================================================================

WITH base AS (
    -- Apólices encerradas antes de dt_ref — point-in-time para sinistros/severidade
    SELECT
        mun,
        apolice,
        dt_inicio_vigencia,
        dt_fim_vigencia,
        evento,
        total_seg,
        indenizacao
    FROM 02_silver.seg_rural.seg_cleaned
    WHERE dt_fim_vigencia < '{dt_ref}'
),
historico_agg AS (
    SELECT
        mun,

        -- ── Janela 90 dias ───────────────────────────────────────────────────
        COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 90)
                   THEN apolice END)
            AS nrApolicesMun90d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 90)
                 AND evento != 'nenhum' THEN 1 ELSE 0 END)
            AS nrSinistrosMun90d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 90)
                 AND evento != 'nenhum' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 90)
                            THEN apolice END), 0)
            AS nrTaxaSinistroMun90d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 90)
                 THEN indenizacao ELSE 0 END)
        / NULLIF(SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 90)
                          THEN total_seg ELSE 0 END), 0)
            AS nrIndiceSeveridadeMun90d,

        -- ── Janela 365 dias ──────────────────────────────────────────────────
        COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
                   THEN apolice END)
            AS nrApolicesMun365d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
                 AND evento != 'nenhum' THEN 1 ELSE 0 END)
            AS nrSinistrosMun365d,

        -- Taxa de sinistro (% de apólices acionadas no município)
        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
                 AND evento != 'nenhum' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
                            THEN apolice END), 0)
            AS nrTaxaSinistroMun365d,

        -- Índice de severidade: indenização paga / valor total segurado
        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
                 THEN indenizacao ELSE 0 END)
        / NULLIF(SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
                          THEN total_seg ELSE 0 END), 0)
            AS nrIndiceSeveridadeMun365d,

        -- ── Janela 730 dias (2 anos) ─────────────────────────────────────────
        COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730)
                   THEN apolice END)
            AS nrApolicesMun730d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730)
                 AND evento != 'nenhum' THEN 1 ELSE 0 END)
            AS nrSinistrosMun730d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730)
                 AND evento != 'nenhum' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730)
                            THEN apolice END), 0)
            AS nrTaxaSinistroMun730d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730)
                 THEN indenizacao ELSE 0 END)
        / NULLIF(SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 730)
                          THEN total_seg ELSE 0 END), 0)
            AS nrIndiceSeveridadeMun730d,

        -- ── Janela 1095 dias (3 anos) ────────────────────────────────────────
        COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 1095)
                   THEN apolice END)
            AS nrApolicesMun1095d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 1095)
                 AND evento != 'nenhum' THEN 1 ELSE 0 END)
            AS nrSinistrosMun1095d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 1095)
                 AND evento != 'nenhum' THEN 1 ELSE 0 END)
        / NULLIF(COUNT(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 1095)
                            THEN apolice END), 0)
            AS nrTaxaSinistroMun1095d,

        SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 1095)
                 THEN indenizacao ELSE 0 END)
        / NULLIF(SUM(CASE WHEN dt_fim_vigencia >= DATE_SUB('{dt_ref}', 1095)
                          THEN total_seg ELSE 0 END), 0)
            AS nrIndiceSeveridadeMun1095d

    FROM base
    GROUP BY mun
),
aberturas_agg AS (
    -- Apólices abertas recentemente (inclui ainda ativas) — proxy de seleção adversa
    SELECT
        mun,
        COUNT(CASE WHEN dt_inicio_vigencia >= DATE_SUB('{dt_ref}', 30)
                   THEN apolice END) AS nrApolicesAbertas30d,
        COUNT(CASE WHEN dt_inicio_vigencia >= DATE_SUB('{dt_ref}', 90)
                   THEN apolice END) AS nrApolicesAbertas90d
    FROM 02_silver.seg_rural.seg_cleaned
    WHERE dt_inicio_vigencia >= DATE_SUB('{dt_ref}', 90)
      AND dt_inicio_vigencia <  '{dt_ref}'
    GROUP BY mun
),
evento_dominante AS (
    -- Evento mais frequente por município nos últimos 365 dias (sinistros encerrados)
    SELECT
        mun,
        evento,
        ROW_NUMBER() OVER (
            PARTITION BY mun
            ORDER BY COUNT(*) DESC
        ) AS rn
    FROM base
    WHERE dt_fim_vigencia >= DATE_SUB('{dt_ref}', 365)
      AND evento != 'nenhum'
    GROUP BY mun, evento
)

SELECT
    CAST('{dt_ref}' AS DATE)              AS dtRef,
    h.mun,

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
    COALESCE(a.nrApolicesAbertas30d, 0)   AS nrApolicesAbertas30d,
    COALESCE(a.nrApolicesAbertas90d, 0)   AS nrApolicesAbertas90d,

    -- ── Evento dominante nos últimos 365 dias (categórica para encoding) ──────
    ed.evento                             AS nrEventosDominante365d

FROM historico_agg h
LEFT JOIN aberturas_agg a  ON a.mun  = h.mun
LEFT JOIN evento_dominante ed ON ed.mun = h.mun AND ed.rn = 1
