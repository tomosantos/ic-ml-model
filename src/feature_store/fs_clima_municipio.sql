-- =============================================================================
-- fs_clima_municipio.sql
-- Feature Store — Indicadores climáticos e de mercado por município
--
-- Chave de entidade : mun  (código IBGE do município)
-- Referência temporal: '{dt_ref}' (parâmetro externo — primeiro dia de um mês)
--
-- Fontes de dados:
--   ONI   → 00_raw.clima.oni_mensal               (série NOAA/CPC)
--   SPI   → 00_raw.clima.spi_municipio_mensal      (calculado via ERA5)
--   Preços → 00_raw.mercado.precos_commodity_mensal (CEPEA/ESALQ)
--
-- Convenção ONI:
--   dtRef = DATE_TRUNC('MONTH', dt_inicio_vigencia) →  ano/mês extraídos para join.
--   O ONI de referência é o valor do mês central da janela sazonal em que
--   dt_ref se enquadra (ex.: apólice em março 2023 → ONI FMA 2023).
--
-- Convenção SPI:
--   Usei a precipitação acumulada dos 3 e 6 meses anteriores a dt_ref,
--   portanto join por (mun, ano(dt_ref), mes(dt_ref)) é point-in-time correto.
--
-- Convenção Preços:
--   Preço mensal médio da saca (R$/sc 60kg) no mês de dt_ref.
--   Quatro colunas fixas (soja, milho 1a safra, milho 2a safra, café) para
--   facilitar o FeatureLookup; no treino, o pipeline seleciona a cultura
--   aplicável de acordo com a apólice. NULL indica preço não disponível.
-- =============================================================================

WITH dtref_parts AS (
    -- Extrai ano e mês de dt_ref uma única vez para reutilizar nos joins
    SELECT
        YEAR('{dt_ref}')  AS ano_ref,
        MONTH('{dt_ref}') AS mes_ref
),

features_mun AS (
    -- Municípios únicos da Silver disponíveis até dt_ref (point-in-time)
    SELECT DISTINCT
        mun
    FROM 02_silver.seg_rural.seg_cleaned
    WHERE dt_inicio_vigencia < '{dt_ref}'
),

oni_ref AS (
    -- ONI do mês corrente de dt_ref (valor mensal central da janela ENSO)
    SELECT
        d.ano_ref,
        d.mes_ref,
        o.oni_valor   AS nrOniValor,
        o.fase        AS flFaseOni
    FROM dtref_parts d
    LEFT JOIN 00_raw.clima.oni_mensal o
        ON  o.ano = d.ano_ref
        AND o.mes = d.mes_ref
),

spi_ref AS (
    -- SPI-3 e SPI-6 por município no mês de dt_ref
    SELECT
        s.mun,
        s.spi_3m  AS nrSpi3m,
        s.spi_6m  AS nrSpi6m
    FROM dtref_parts d
    JOIN 00_raw.clima.spi_municipio_mensal s
        ON  s.ano = d.ano_ref
        AND s.mes = d.mes_ref
),

precos_ref AS (
    -- Preços mensais das quatro culturas principais no mês de dt_ref
    -- (preço nacional — repetido para todos os municípios)
    SELECT
        d.ano_ref,
        d.mes_ref,
        MAX(CASE WHEN p.cultura = 'soja'           THEN p.preco_rs_saca END) AS nrPrecoSoja,
        MAX(CASE WHEN p.cultura = 'milho_1a_safra' THEN p.preco_rs_saca END) AS nrPrecoMilho1a,
        MAX(CASE WHEN p.cultura = 'milho_2a_safra' THEN p.preco_rs_saca END) AS nrPrecoMilho2a,
        MAX(CASE WHEN p.cultura = 'cafe'           THEN p.preco_rs_saca END) AS nrPrecoCafe
    FROM dtref_parts d
    LEFT JOIN 00_raw.mercado.precos_commodity_mensal p
        ON  p.ano = d.ano_ref
        AND p.mes = d.mes_ref
    GROUP BY d.ano_ref, d.mes_ref
)

SELECT
    -- ── Chaves primárias ──────────────────────────────────────────────────────
    CAST('{dt_ref}' AS DATE)          AS dtRef,
    m.mun,

    -- ── ONI — Ciclo ENSO ─────────────────────────────────────────────────────
    -- nrOniValor: anomalia da temperatura da superfície do Pacífico equatorial
    --   (positivo = El Niño, negativo = La Niña)
    o.nrOniValor,

    -- flFaseOni: fase categórica do ENSO (el_nino / la_nina / neutro)
    --   Threshold ±0.5 conforme NOAA; codificação ordinal aplicada no treino.
    o.flFaseOni,

    -- ── SPI — Índice de Seca por município ───────────────────────────────────
    -- nrSpi3m: SPI acumulado 3 meses (condição climática trimestral recente)
    --   < -1.5 → seca severa; > +1.5 → excesso hídrico severo
    s.nrSpi3m,

    -- nrSpi6m: SPI acumulado 6 meses (condição climática semestral)
    --   Melhor proxy para déficit hídrico persistente que impacta culturas
    --   de ciclo longo (café, cana, pecuária)
    s.nrSpi6m,

    -- ── Preços de Commodity — Mercado ────────────────────────────────────────
    -- Preço nacional médio mensal (R$/sc 60 kg) por cultura CEPEA/ESALQ.
    -- NULL quando a série não cobre o mês de dtRef.
    -- No pipeline de treino, o valor aplicável é selecionado pela cultura
    -- da apólice usando CASE WHEN ou FeatureLookup por cultura.
    p.nrPrecoSoja,
    p.nrPrecoMilho1a,
    p.nrPrecoMilho2a,
    p.nrPrecoCafe

FROM features_mun       m
CROSS JOIN oni_ref      o
LEFT JOIN  spi_ref      s ON s.mun = m.mun
CROSS JOIN precos_ref   p
