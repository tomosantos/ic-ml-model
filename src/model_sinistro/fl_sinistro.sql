-- =============================================================================
-- fl_sinistro.sql
-- Definição da variável resposta (label) para o modelo de sinistro
--
-- A data de referência é o PRIMEIRO DIA DO MÊS de dt_inicio_vigencia.
-- Convenção: dtRef = DATE_TRUNC('MONTH', dt_inicio_vigencia)
-- Isso alinha o join com as Feature Store tables (fs_apolice_financeiro,
-- fs_historico_municipio, fs_risco_cultura_uf), que também usam granularidade
-- mensal — evitando features nulas silenciosas no FeatureLookup.
--
-- Apenas apólices com ciclo completo (dt_fim_vigencia < current_date) são
-- incluídas para garantir que o desfecho seja definitivo no treino.
-- =============================================================================

WITH tb_labels AS (
    SELECT
        apolice,

        -- Referência temporal para lookup na Feature Store
        -- dtRef = primeiro dia do mês de início de vigência (granularidade mensal)
        DATE_TRUNC('MONTH', dt_inicio_vigencia)         AS dtRef,

        -- ── Variável resposta principal (classificação binária) ───────────────
        CASE
            WHEN evento != 'nenhum' THEN 1
            ELSE 0
        END                                             AS flSinistro,

        -- ── Variáveis auxiliares de severidade ───────────────────────────────
        evento                                          AS dsEvento,
        indenizacao                                     AS vlIndenizacaoRealizada,
        sinistralidade                                  AS nrSinistralidade

    FROM 02_silver.seg_rural.seg_cleaned

    -- Somente apólices cujo ciclo já encerrou (desfecho conhecido)
    WHERE dt_fim_vigencia < current_date()
)

SELECT * FROM tb_labels
