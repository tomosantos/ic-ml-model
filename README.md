# ic-ml-model
Insurance Claims ML Model

---

## Dicionário de Dados — Tabela Bronze (`seg_rural`)

| Coluna | Descrição |
|---|---|
| `NM_RAZAO_SOCIAL` | Razão social da seguradora |
| `CD_PROCESSO_SUSEP` | Código do produto registrado na SUSEP |
| `NR_PROPOSTA` | Número da proposta na seguradora |
| `ID_PROPOSTA` | Código identificador da proposta no sistema do MAPA (SISSER) |
| `DT_PROPOSTA` | Data da contratação da proposta |
| `DT_INICIO_VIGENCIA` | Data de início da vigência do seguro |
| `DT_FIM_VIGENCIA` | Data do fim da vigência do seguro |
| `NM_SEGURADO` | Nome do segurado |
| `NR_DOCUMENTO_SEGURADO` | Número do CPF ou CNPJ do segurado |
| `NM_MUNICIPIO_PROPRIEDADE` | Nome do município onde está localizada a propriedade |
| `SG_UF_PROPRIEDADE` | Sigla da Unidade da Federação onde está localizada a propriedade |
| `LATITUDE` | Latitude da propriedade |
| `NR_GRAU_LAT` | Grau da latitude da propriedade |
| `NR_MIN_LAT` | Minuto da latitude da propriedade |
| `NR_SEG_LAT` | Segundo da latitude da propriedade |
| `LONGITUDE` | Longitude da propriedade |
| `NR_GRAU_LONG` | Grau da longitude da propriedade |
| `NR_MIN_LONG` | Minuto da longitude da propriedade |
| `NR_SEG_LONG` | Segundo da longitude da propriedade |
| `NM_CLASSIF_PRODUTO` | Classificação do tipo de seguro |
| `NM_CULTURA_GLOBAL` | Cultura ou atividade segurada |
| `NR_AREA_TOTAL` | Área total segurada |
| `NR_ANIMAL` | Número de animais segurados |
| `NR_PRODUTIVIDADE_ESTIMADA` | Produtividade estimada |
| `NR_PRODUTIVIDADE_SEGURADA` | Produtividade segurada |
| `NivelDeCobertura` | Nível de cobertura do seguro |
| `VL_LIMITE_GARANTIA` | Valor segurado |
| `VL_PREMIO_LIQUIDO` | Valor do prêmio |
| `PE_TAXA` | Percentual da taxa de prêmio |
| `VL_SUBVENCAO_FEDERAL` | Valor da subvenção federal |
| `NR_APOLICE` | Número da apólice na seguradora |
| `DT_APOLICE` | Data de contratação da apólice |
| `ANO_APOLICE` | Ano de contratação da apólice |
| `CD_GEOCMU` | Geocódigo do município onde está localizada a propriedade |
| `VALOR_INDENIZAÇÃO` | Valor pago em indenização, em caso de sinistro |
| `EVENTO_PREPONDERANTE` | Evento preponderante causador do sinistro |

### Mapeamento Bronze → Silver

Colunas renomeadas e derivadas na camada Silver (`seg_rural.seg_cleaned`):

| Coluna Bronze | Coluna Silver | Descrição |
|---|---|---|
| `NM_RAZAO_SOCIAL` | `seguradora` | Razão social da seguradora |
| `NM_MUNICIPIO_PROPRIEDADE` | `nome_mun` | Nome do município da propriedade |
| `SG_UF_PROPRIEDADE` | `uf` | Sigla da UF da propriedade |
| `NM_CLASSIF_PRODUTO` | `tipo` | Tipo de seguro (`custeio`, `produtividade`, etc.) |
| `NM_CULTURA_GLOBAL` | `cultura` | Cultura ou atividade segurada |
| `NR_AREA_TOTAL` | `area` | Área total segurada |
| `NR_ANIMAL` | `animal` | Número de animais segurados |
| `NR_PRODUTIVIDADE_ESTIMADA` | `prod_est` | Produtividade estimada |
| `NR_PRODUTIVIDADE_SEGURADA` | `prod_seg` | Produtividade segurada |
| `NivelDeCobertura` | `nivel_cob` | Nível de cobertura do seguro |
| `VL_LIMITE_GARANTIA` | `total_seg` | Valor segurado |
| `VL_PREMIO_LIQUIDO` | `premio` | Valor do prêmio |
| `PE_TAXA` | `taxa` | Percentual da taxa de prêmio |
| `VL_SUBVENCAO_FEDERAL` | `subvencao` | Valor da subvenção federal |
| `NR_APOLICE` | `apolice` | Número da apólice |
| `CD_GEOCMU` | `mun` | Código IBGE do município |
| `VALOR_INDENIZAÇÃO` | `indenizacao` | Valor pago em indenização |
| `EVENTO_PREPONDERANTE` | `evento` | Evento causador do sinistro (normalizado) |
| *(derivada)* | `duracao` | Dias entre início e fim de vigência |
| *(derivada)* | `tipo_cultura` | Categoria da cultura (`graos`, `frutas`, etc.) |
| *(derivada)* | `sinistro` | Flag de sinistro: `0` = sem evento, `1` = com evento |
| *(derivada)* | `sinistralidade` | Razão `indenizacao / premio` |
| *(derivada)* | `regiao` | Região geográfica do Brasil |
