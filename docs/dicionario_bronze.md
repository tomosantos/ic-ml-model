# Dicionário de Dados — Tabela Bronze (`seg_rural`)

Esta tabela contém os dados brutos ingeridos do SISSER (MAPA/PSR), sem transformações analíticas, mantendo as nomenclaturas originais do arquivo público.

| Coluna | Tipo Esperado | Descrição |
|---|---|---|
| `NM_RAZAO_SOCIAL` | String | Razão social da seguradora |
| `CD_PROCESSO_SUSEP` | String | Código do produto registrado na SUSEP |
| `NR_PROPOSTA` | String | Número da proposta na seguradora |
| `ID_PROPOSTA` | String | Código identificador da proposta no sistema do MAPA (SISSER) |
| `DT_PROPOSTA` | Date | Data da contratação da proposta |
| `DT_INICIO_VIGENCIA` | Date | Data de início da vigência do seguro |
| `DT_FIM_VIGENCIA` | Date | Data do fim da vigência do seguro |
| `NM_SEGURADO` | String | Nome do segurado |
| `NR_DOCUMENTO_SEGURADO` | String | Número do CPF ou CNPJ do segurado |
| `NM_MUNICIPIO_PROPRIEDADE` | String | Nome do município onde está localizada a propriedade |
| `SG_UF_PROPRIEDADE` | String | Sigla da Unidade da Federação onde está localizada a propriedade |
| `LATITUDE` | Double | Latitude da propriedade |
| `NR_GRAU_LAT` | Int | Grau da latitude da propriedade |
| `NR_MIN_LAT` | Int | Minuto da latitude da propriedade |
| `NR_SEG_LAT` | Double | Segundo da latitude da propriedade |
| `LONGITUDE` | Double | Longitude da propriedade |
| `NR_GRAU_LONG` | Int | Grau da longitude da propriedade |
| `NR_MIN_LONG` | Int | Minuto da longitude da propriedade |
| `NR_SEG_LONG` | Double | Segundo da longitude da propriedade |
| `NM_CLASSIF_PRODUTO` | String | Classificação do tipo de seguro |
| `NM_CULTURA_GLOBAL` | String | Cultura ou atividade segurada |
| `NR_AREA_TOTAL` | Double | Área total segurada |
| `NR_ANIMAL` | Double | Número de animais segurados |
| `NR_PRODUTIVIDADE_ESTIMADA` | Double | Produtividade estimada |
| `NR_PRODUTIVIDADE_SEGURADA` | Double | Produtividade segurada |
| `NivelDeCobertura` | Double | Nível de cobertura do seguro |
| `VL_LIMITE_GARANTIA` | Double | Valor segurado |
| `VL_PREMIO_LIQUIDO` | Double | Valor do prêmio |
| `PE_TAXA` | Double | Percentual da taxa de prêmio |
| `VL_SUBVENCAO_FEDERAL` | Double | Valor da subvenção federal |
| `NR_APOLICE` | String | Número da apólice na seguradora |
| `DT_APOLICE` | Date | Data de contratação da apólice |
| `ANO_APOLICE` | Int | Ano de contratação da apólice |
| `CD_GEOCMU` | String | Geocódigo do município onde está localizada a propriedade |
| `VALOR_INDENIZAÇÃO` | Double | Valor pago em indenização, em caso de sinistro |
| `EVENTO_PREPONDERANTE` | String | Evento preponderante causador do sinistro |
