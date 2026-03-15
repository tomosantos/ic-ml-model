# Dicionário de Dados — Tabela Silver (`seg_cleaned`)

Esta tabela representa a camada Silver (`02_silver.seg_rural.seg_cleaned`), que contém dados consolidados e normalizados. Colunas sensíveis ou metadados de localização detalhada foram removidos em relação à Bronze.

| Coluna | Tipo Esperado | Origem Bronze | Descrição |
|---|---|---|---|
| `seguradora` | String | `NM_RAZAO_SOCIAL` | Razão social da seguradora |
| `nome_mun` | String | `NM_MUNICIPIO_PROPRIEDADE` | Nome do município da propriedade (normalizado) |
| `uf` | String | `SG_UF_PROPRIEDADE` | Sigla da UF da propriedade |
| `tipo` | String | `NM_CLASSIF_PRODUTO` | Tipo de seguro (ex: `custeio`, `produtividade`) |
| `cultura` | String | `NM_CULTURA_GLOBAL` | Cultura ou atividade segurada |
| `area` | Double | `NR_AREA_TOTAL` | Área total segurada |
| `animal` | Double | `NR_ANIMAL` | Número de animais segurados |
| `prod_est` | Double | `NR_PRODUTIVIDADE_ESTIMADA` | Produtividade estimada |
| `prod_seg` | Double | `NR_PRODUTIVIDADE_SEGURADA` | Produtividade segurada |
| `nivel_cob` | Double | `NivelDeCobertura` | Nível de cobertura do seguro (0 a 1) |
| `total_seg` | Double | `VL_LIMITE_GARANTIA` | Valor máximo de limite de garantia/valor segurado |
| `premio` | Double | `VL_PREMIO_LIQUIDO` | Valor do prêmio pago |
| `taxa` | Double | `PE_TAXA` | Percentual da taxa de prêmio |
| `subvencao` | Double | `VL_SUBVENCAO_FEDERAL` | Valor da subvenção federal recebida |
| `apolice` | String | `NR_APOLICE` | Número da apólice |
| `dt_inicio_vigencia` | Date | `DT_INICIO_VIGENCIA` | Data de início de vigência de cobertura securitária |
| `dt_fim_vigencia` | Date | `DT_FIM_VIGENCIA` | Data de término da cobertura securitária |
| `mun` | String | `CD_GEOCMU` | Código IBGE do município (7 dígitos) |
| `indenizacao` | Double | `VALOR_INDENIZAÇÃO` | Valor pago em indenização |
| `evento` | String | `EVENTO_PREPONDERANTE` | Causa raiz mapeada para o sinistro |
| `duracao` | Int | *(derivada)* | Dias absolutos informados de vigência (Fim - Início) |
| `tipo_cultura` | String | *(derivada)* | Categoria da cultura (`graos`, `frutas`, `leguminosas`, etc.) |
| `sinistro` | Int | *(derivada)* | Flag de sinistro: `0` = sem evento, `1` = com evento (Label/Target) |
| `sinistralidade` | Double | *(derivada)* | Razão `indenizacao / premio` |
| `regiao` | String | *(derivada)* | Região geográfica do Brasil mapeada pela `uf` |
