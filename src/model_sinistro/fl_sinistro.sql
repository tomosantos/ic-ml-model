-- =============================================================================
-- fl_sinistro.sql
-- Definição da variável resposta (label) para o modelo de sinistro
--
-- A data de referência é dt_inicio_vigencia: features históricas devem ser
-- calculadas olhando para o passado a partir DESTA data.
--
-- Apenas apólices com ciclo completo (dt_fim_vigencia < current_date) são
-- incluídas para garantir que o desfecho seja definitivo no treino.
-- =============================================================================

WITH tb_labels AS (
    SELECT
        apolice,

        -- Referência temporal para lookup na Feature Store
        dt_inicio_vigencia                              AS dtRef,

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
