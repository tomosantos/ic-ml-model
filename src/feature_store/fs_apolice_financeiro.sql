-- =============================================================================
-- fs_apolice_financeiro.sql
-- Feature Store — Engenharia financeira da apólice
--
-- Chave de entidade : apolice
-- Referência temporal: dt_ref = DATE_TRUNC('MONTH', dt_inicio_vigencia)
--
-- Convenção: dtRef é sempre o primeiro dia do mês de início da vigência.
-- Isso garante alinhamento com fs_historico_municipio e fs_risco_cultura_uf,
-- que também operam em granularidade mensal, permitindo FeatureLookup correto.
--
-- Features derivadas exclusivamente de informações disponíveis NO ATO da
-- contratação — sem qualquer informação pós-contratual.
-- =============================================================================

SELECT
    apolice,
    DATE_TRUNC('MONTH', dt_inicio_vigencia)     AS dtRef,

    -- ── Features temporais ───────────────────────────────────────────────────
    -- Mês de plantio codificado numericamente (1–12); a codificação cíclica
    -- seno/cosseno é aplicada no pipeline de treino para evitar ruptura
    -- entre dezembro e janeiro.
    MONTH(dt_inicio_vigencia)                   AS nrMesPlantio,
    YEAR(dt_inicio_vigencia)                    AS nrAnoPlantio,
    duracao                                     AS nrDuracaoDias,

    -- ── Densidade e normalização financeira ──────────────────────────────────
    -- Valor segurado por hectare (normaliza tamanho da propriedade)
    total_seg / NULLIF(area, 0)                 AS nrDensidadeValorSegHa,

    -- Prêmio por hectare
    premio    / NULLIF(area, 0)                 AS nrPremioPorHa,

    -- ── Apetite ao risco / risco moral ───────────────────────────────────────
    -- Razão produtividade segurada / estimada: quanto do risco o produtor
    -- transferiu à seguradora (1.0 = cobertura total)
    prod_seg  / NULLIF(prod_est, 0)             AS nrRazaoCoberturaProd,

    -- ── Precificação e subsídio ───────────────────────────────────────────────
    -- Proporção do prêmio coberta pela subvenção federal
    subvencao / NULLIF(premio, 0)               AS nrProporcaoSubvencao,

    -- Taxa da apólice como proxy do risco já precificado pela seguradora
    taxa                                        AS nrTaxaApolice,

    -- Nível de cobertura contratado (0–1)
    nivel_cob                                   AS nrNivelCobertura

FROM 02_silver.seg_rural.seg_cleaned
