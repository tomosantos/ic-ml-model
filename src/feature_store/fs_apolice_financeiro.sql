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
    QUARTER(dt_inicio_vigencia)                 AS nrTrimestre,
    YEAR(dt_inicio_vigencia)                    AS nrAnoPlantio,
    duracao                                     AS nrDuracaoDias,
    -- Duração normalizada para facilitar comparação entre culturas de ciclo
    -- curto e longo (1.0 = apólice anual inteira).
    duracao / 365.0                             AS nrDuracaoRelativa,
    -- Indicador de safra verão (grãos): meses de set–jan, período em que
    -- o risco climático difere significativamente da safra inverno/segunda.
    CASE
        WHEN MONTH(dt_inicio_vigencia) >= 9
          OR MONTH(dt_inicio_vigencia) = 1 THEN 1
        ELSE 0
    END                                         AS flSafraVerao,

    -- ── Densidade e normalização financeira ──────────────────────────────────
    -- Valor segurado por hectare: indica o patrimônio em risco por unidade
    -- de área — conceito distinto de nrPremioPorHa (custo do seguro/ha).
    total_seg / NULLIF(area, 0)                 AS nrDensidadeValorSegHa,

    -- Custo do seguro por hectare: normaliza a exposição financeira pelo
    -- tamanho da propriedade; correlaciona com intensidade de uso da terra.
    premio    / NULLIF(area, 0)                 AS nrPremioPorHa,

    -- ── Apetite ao risco / risco moral ───────────────────────────────────────
    -- Razão produtividade segurada / estimada: quanto do risco o produtor
    -- transferiu à seguradora (1.0 = cobertura total)
    prod_seg  / NULLIF(prod_est, 0)             AS nrRazaoCoberturaProd,

    -- ── Precificação e subsídio ───────────────────────────────────────────────
    -- Razão subvenção/prêmio: mede dependência do produtor em subsídio federal;
    -- alta dependência pode correlacionar com perfil de risco elevado.
    subvencao / NULLIF(premio, 0)               AS nrRazaoSubvencaoPremio,

    -- Taxa da apólice como proxy do risco já precificado pela seguradora
    taxa                                        AS nrTaxaApolice,

    -- Nível de cobertura contratado (0–1)
    nivel_cob                                   AS nrNivelCobertura,

    -- ── Pecuária ─────────────────────────────────────────────────────────────
    -- Intensidade de uso do pasto: área por cabeça de animal segurado.
    -- NULL para apólices não-pecuárias (animal = 0 ou NULL).
    CASE
        WHEN animal > 0
        THEN area / NULLIF(animal, 0)
    END                                         AS nrAreaPorAnimal

FROM 02_silver.seg_rural.seg_cleaned
-- Garante unicidade da chave (dtRef, apolice): em caso de registros duplicados
-- na camada Silver (re-ingestão, emendas), mantém o primeiro por data de início.
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY apolice, DATE_TRUNC('MONTH', dt_inicio_vigencia)
    ORDER BY     dt_inicio_vigencia
) = 1
