# Databricks notebook source
# DBTITLE 1,Setup
import sys
import re
import unicodedata
import pandas as pd
from itertools import chain
from pyspark.sql import functions as F
from pyspark.sql.functions import to_date, datediff, create_map, split, array_join
from pyspark.sql.types import StringType, IntegerType, DoubleType

sys.path.insert(0, '../lib')
from const import (
    TABLE_BRONZE_HISTORICAL,
    TABLE_BRONZE_ATUAL,
    ACENTOS,
    SEM_ACENTOS,
    COLUNAS_RETIRAR,
    RENAME_MAP,
    TIPO_MAP,
    EVENTO_MAP,
    TIPO_CULTURA_MAP,
    COLUNAS_FINAIS,
    REPLACERS_MUN,
    IBGE_CSV_URL,
    UF_MAP,
    DISTRITOS_PATH,
    REPLACERS_SEG,
    REGIAO_MAP,
    TABLE_SILVER_CLEANED
)


# COMMAND ----------

# DBTITLE 1,Verificação das Colunas
df_historical = spark.read.table(TABLE_BRONZE_HISTORICAL)
df_actual = spark.read.table(TABLE_BRONZE_ATUAL)

df_historical.columns == df_actual.columns

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE 01_bronze.seg_rural.historical_seg

# COMMAND ----------

# DBTITLE 1,Criação da Tabela Silver
# MAGIC %sql
# MAGIC -- 1. Configuração de Contexto
# MAGIC USE CATALOG 02_silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS 02_silver.seg_rural;
# MAGIC USE SCHEMA seg_rural;
# MAGIC
# MAGIC -- 2. Garantir que a tabela de destino existe
# MAGIC -- Criamos a estrutura baseada na união das tabelas da Bronze (vazia)
# MAGIC CREATE TABLE IF NOT EXISTS 02_silver.seg_rural.seg_cleaned
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT * FROM 01_bronze.seg_rural.historical_seg WHERE 1=0;
# MAGIC
# MAGIC -- 3. Preparação dos Dados (Origem)
# MAGIC CREATE OR REPLACE TEMP VIEW agg_seg AS 
# MAGIC WITH raw AS (
# MAGIC   SELECT
# MAGIC     CAST(NM_RAZAO_SOCIAL AS STRING) AS NM_RAZAO_SOCIAL,
# MAGIC     CAST(CD_PROCESSO_SUSEP AS STRING) AS CD_PROCESSO_SUSEP,
# MAGIC     CAST(NR_PROPOSTA AS STRING) AS NR_PROPOSTA,
# MAGIC     CAST(ID_PROPOSTA AS STRING) AS ID_PROPOSTA,
# MAGIC     DT_PROPOSTA,
# MAGIC     DT_INICIO_VIGENCIA,
# MAGIC     DT_FIM_VIGENCIA,
# MAGIC     CAST(NM_SEGURADO AS STRING) AS NM_SEGURADO,
# MAGIC     CAST(NR_DOCUMENTO_SEGURADO AS STRING) AS NR_DOCUMENTO_SEGURADO,
# MAGIC     CAST(NM_MUNICIPIO_PROPRIEDADE AS STRING) AS NM_MUNICIPIO_PROPRIEDADE,
# MAGIC     CAST(SG_UF_PROPRIEDADE AS STRING) AS SG_UF_PROPRIEDADE,
# MAGIC     CAST(LATITUDE AS STRING) AS LATITUDE,
# MAGIC     CAST(NR_GRAU_LAT AS STRING) AS NR_GRAU_LAT,
# MAGIC     CAST(NR_MIN_LAT AS STRING) AS NR_MIN_LAT,
# MAGIC     CAST(NR_SEG_LAT AS STRING) AS NR_SEG_LAT,
# MAGIC     CAST(LONGITUDE AS STRING) AS LONGITUDE,
# MAGIC     CAST(NR_GRAU_LONG AS STRING) AS NR_GRAU_LONG,
# MAGIC     CAST(NR_MIN_LONG AS STRING) AS NR_MIN_LONG,
# MAGIC     CAST(NR_SEG_LONG AS STRING) AS NR_SEG_LONG,
# MAGIC     CAST(NR_DECIMAL_LATITUDE AS STRING) AS NR_DECIMAL_LATITUDE,
# MAGIC     CAST(NR_DECIMAL_LONGITUDE AS STRING) AS NR_DECIMAL_LONGITUDE,
# MAGIC     CAST(NM_CLASSIF_PRODUTO AS STRING) AS NM_CLASSIF_PRODUTO,
# MAGIC     CAST(NM_CULTURA_GLOBAL AS STRING) AS NM_CULTURA_GLOBAL,
# MAGIC     CAST(NR_AREA_TOTAL AS STRING) AS NR_AREA_TOTAL,
# MAGIC     CAST(NR_ANIMAL AS STRING) AS NR_ANIMAL,
# MAGIC     CAST(NR_PRODUTIVIDADE_ESTIMADA AS STRING) AS NR_PRODUTIVIDADE_ESTIMADA,
# MAGIC     CAST(NR_PRODUTIVIDADE_SEGURADA AS STRING) AS NR_PRODUTIVIDADE_SEGURADA,
# MAGIC     CAST(NivelDeCobertura AS STRING) AS NivelDeCobertura,
# MAGIC     CAST(VL_LIMITE_GARANTIA AS STRING) AS VL_LIMITE_GARANTIA,
# MAGIC     CAST(VL_PREMIO_LIQUIDO AS STRING) AS VL_PREMIO_LIQUIDO,
# MAGIC     CAST(PE_TAXA AS STRING) AS PE_TAXA,
# MAGIC     CAST(VL_SUBVENCAO_FEDERAL AS STRING) AS VL_SUBVENCAO_FEDERAL,
# MAGIC     CAST(NR_APOLICE AS STRING) AS NR_APOLICE,
# MAGIC     DT_APOLICE,
# MAGIC     CAST(ANO_APOLICE AS STRING) AS ANO_APOLICE,
# MAGIC     CAST(CD_GEOCMU AS STRING) AS CD_GEOCMU,
# MAGIC     CAST(`VALOR_INDENIZAÇÃO` AS STRING) AS `VALOR_INDENIZAÇÃO`,
# MAGIC     CAST(EVENTO_PREPONDERANTE AS STRING) AS EVENTO_PREPONDERANTE
# MAGIC   FROM 01_bronze.seg_rural.historical_seg
# MAGIC   UNION ALL
# MAGIC   SELECT
# MAGIC     CAST(NM_RAZAO_SOCIAL AS STRING) AS NM_RAZAO_SOCIAL,
# MAGIC     CAST(CD_PROCESSO_SUSEP AS STRING) AS CD_PROCESSO_SUSEP,
# MAGIC     CAST(NR_PROPOSTA AS STRING) AS NR_PROPOSTA,
# MAGIC     CAST(ID_PROPOSTA AS STRING) AS ID_PROPOSTA,
# MAGIC     DT_PROPOSTA,
# MAGIC     DT_INICIO_VIGENCIA,
# MAGIC     DT_FIM_VIGENCIA,
# MAGIC     CAST(NM_SEGURADO AS STRING) AS NM_SEGURADO,
# MAGIC     CAST(NR_DOCUMENTO_SEGURADO AS STRING) AS NR_DOCUMENTO_SEGURADO,
# MAGIC     CAST(NM_MUNICIPIO_PROPRIEDADE AS STRING) AS NM_MUNICIPIO_PROPRIEDADE,
# MAGIC     CAST(SG_UF_PROPRIEDADE AS STRING) AS SG_UF_PROPRIEDADE,
# MAGIC     CAST(LATITUDE AS STRING) AS LATITUDE,
# MAGIC     CAST(NR_GRAU_LAT AS STRING) AS NR_GRAU_LAT,
# MAGIC     CAST(NR_MIN_LAT AS STRING) AS NR_MIN_LAT,
# MAGIC     CAST(NR_SEG_LAT AS STRING) AS NR_SEG_LAT,
# MAGIC     CAST(LONGITUDE AS STRING) AS LONGITUDE,
# MAGIC     CAST(NR_GRAU_LONG AS STRING) AS NR_GRAU_LONG,
# MAGIC     CAST(NR_MIN_LONG AS STRING) AS NR_MIN_LONG,
# MAGIC     CAST(NR_SEG_LONG AS STRING) AS NR_SEG_LONG,
# MAGIC     CAST(NR_DECIMAL_LATITUDE AS STRING) AS NR_DECIMAL_LATITUDE,
# MAGIC     CAST(NR_DECIMAL_LONGITUDE AS STRING) AS NR_DECIMAL_LONGITUDE,
# MAGIC     CAST(NM_CLASSIF_PRODUTO AS STRING) AS NM_CLASSIF_PRODUTO,
# MAGIC     CAST(NM_CULTURA_GLOBAL AS STRING) AS NM_CULTURA_GLOBAL,
# MAGIC     CAST(NR_AREA_TOTAL AS STRING) AS NR_AREA_TOTAL,
# MAGIC     CAST(NR_ANIMAL AS STRING) AS NR_ANIMAL,
# MAGIC     CAST(NR_PRODUTIVIDADE_ESTIMADA AS STRING) AS NR_PRODUTIVIDADE_ESTIMADA,
# MAGIC     CAST(NR_PRODUTIVIDADE_SEGURADA AS STRING) AS NR_PRODUTIVIDADE_SEGURADA,
# MAGIC     CAST(NivelDeCobertura AS STRING) AS NivelDeCobertura,
# MAGIC     CAST(VL_LIMITE_GARANTIA AS STRING) AS VL_LIMITE_GARANTIA,
# MAGIC     CAST(VL_PREMIO_LIQUIDO AS STRING) AS VL_PREMIO_LIQUIDO,
# MAGIC     CAST(PE_TAXA AS STRING) AS PE_TAXA,
# MAGIC     CAST(VL_SUBVENCAO_FEDERAL AS STRING) AS VL_SUBVENCAO_FEDERAL,
# MAGIC     CAST(NR_APOLICE AS STRING) AS NR_APOLICE,
# MAGIC     DT_APOLICE,
# MAGIC     CAST(ANO_APOLICE AS STRING) AS ANO_APOLICE,
# MAGIC     CAST(CD_GEOCMU AS STRING) AS CD_GEOCMU,
# MAGIC     CAST(`VALOR_INDENIZAÇÃO` AS STRING) AS `VALOR_INDENIZAÇÃO`,
# MAGIC     CAST(EVENTO_PREPONDERANTE AS STRING) AS EVENTO_PREPONDERANTE
# MAGIC   FROM 01_bronze.seg_rural.seg_2025
# MAGIC ),
# MAGIC dedup AS (
# MAGIC   -- Lógica para garantir que, se um ID aparecer em ambas, pegamos a versão mais recente
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY ID_PROPOSTA ORDER BY DT_PROPOSTA, DT_APOLICE DESC) AS rn
# MAGIC   FROM raw
# MAGIC )
# MAGIC SELECT
# MAGIC   * EXCEPT(rn)
# MAGIC FROM dedup
# MAGIC WHERE rn = 1;

# COMMAND ----------

# DBTITLE 1,Check
# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM agg_seg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tratamento

# COMMAND ----------

# DBTITLE 1,Definindo Funções
def remove_acentos_col(col_name):
  return F.translate(F.col(col_name), ACENTOS, SEM_ACENTOS)

def normalize_str(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')

# COMMAND ----------

# DBTITLE 1,Tratamento - 1
# 1. Captura a View existente
df = spark.table('agg_seg')

# 2. Substituir valores nas colunas
df = df.replace(['-', '...', 'X'], '0')

# 3. Limpa os NOMES das colunas (Removendo acentos e espaços nos cabeçalhos)
novos_nomes = [normalize_str(c) for c in df.columns]

for col in df.columns:
  df = df.withColumnRenamed(col, normalize_str(col))

# 3. Limpa os VALORES das colunas (Apenas colunas do tipo String)
colunas_string = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

for col_name in colunas_string:
  df = df.withColumn(col_name, F.translate(F.col(col_name), ACENTOS, SEM_ACENTOS))


# COMMAND ----------

# DBTITLE 1,Criação da coluna duracao
df = df.withColumn('DT_INICIO_VIGENCIA', to_date(F.col('DT_INICIO_VIGENCIA')))
df = df.withColumn('DT_FIM_VIGENCIA',    to_date(F.col('DT_FIM_VIGENCIA')))
df = df.withColumn('duracao', datediff(F.col('DT_FIM_VIGENCIA'), F.col('DT_INICIO_VIGENCIA')))

# COMMAND ----------

# DBTITLE 1,Limpeza da coluna EVENTO_PREPONDERANTE
# Normaliza espaços internos (equivalente ao str.split().str.join(' ') do pandas)
df = df.withColumn(
    "EVENTO_PREPONDERANTE",
    F.trim(F.regexp_replace(F.col('EVENTO_PREPONDERANTE'), r'[\s\u00a0]+', ' '))
)
df = df.withColumn(
    "EVENTO_PREPONDERANTE",
    F.when(F.col("EVENTO_PREPONDERANTE").isNull(), F.lit("0")).otherwise(
        F.col("EVENTO_PREPONDERANTE")
    ),
)

# COMMAND ----------

# DBTITLE 1,Remoção de espaços em colunas de texto
colunas_string = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]

for col_name in colunas_string:
  df = df.withColumn(col_name, F.trim(F.col(col_name)))

# COMMAND ----------

# DBTITLE 1,Remoção de colunas desnecessárias
# Retira apenas as colunas que realmente existem no DataFrame
colunas_retirar = [c for c in COLUNAS_RETIRAR if c in df.columns]
df = df.drop(*colunas_retirar)

# COMMAND ----------

# DBTITLE 1,Renomear colunas para nomes curtos/padronizados
for old, new in RENAME_MAP.items():
  if old in df.columns:
    df = df.withColumnRenamed(old, new)

# COMMAND ----------

# DBTITLE 1,Preenchimento de nulos e conversão de tipos
# Preenche nulos em colunas string com '0'
colunas_string = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
df = df.fillna('0', subset=colunas_string)

# Preenche nulos em colunas numéricas com 0
colunas_numericas = [
  f.name for f in df.schema.fields
  if isinstance(f.dataType, (IntegerType, DoubleType))
]
df = df.fillna(0, subset=colunas_numericas)

# Coluna 'animal': substitui marcadores residuais e converte para inteiro
if 'animal' in df.columns:
  df = df.withColumn('animal',
    F.when(F.col('animal').isin('-', '...', 'X', '0'), F.lit(0))
     .otherwise(F.col('animal'))
     .cast('int')
  )

# Coluna 'mun': converte para inteiro
if 'mun' in df.columns:
  df = df.withColumn('mun', F.col('mun').cast('int'))

# Colunas numéricas contínuas: converte para double (necessário para operações aritméticas)
colunas_double = ['area', 'prod_est', 'prod_seg', 'nivel_cob', 'total_seg',
                  'premio', 'taxa', 'subvencao', 'indenizacao', 'lat', 'lon']
for col_name in colunas_double:
  if col_name in df.columns:
    df = df.withColumn(col_name, F.col(col_name).cast('double'))

# COMMAND ----------

# DBTITLE 1,Mapeamento da coluna tipo
tipo_expr = create_map([F.lit(x) for x in chain(*TIPO_MAP.items())])
df = df.withColumn('tipo', F.coalesce(tipo_expr[F.col('tipo')], F.col('tipo')))

# COMMAND ----------

# DBTITLE 1,Mapeamento da coluna evento
evento_expr = create_map([F.lit(x) for x in chain(*EVENTO_MAP.items())])
df = df.withColumn('evento', F.coalesce(evento_expr[F.col('evento')], F.col('evento')))

# COMMAND ----------

# DBTITLE 1,Criação da coluna tipo_cultura
tipo_cultura_expr = create_map([F.lit(x) for x in chain(*TIPO_CULTURA_MAP.items())])
df = df.withColumn(
  'tipo_cultura',
  F.coalesce(tipo_cultura_expr[F.col('cultura')], F.lit('outros'))
)

# COMMAND ----------

# DBTITLE 1,Criação das colunas sinistro e sinistralidade
# sinistro: 0 se não houve evento (valor numérico "nenhum"), 1 caso contrário
df = df.withColumn(
  'sinistro',
  F.when(F.col('evento') == 'nenhum', F.lit(0)).otherwise(F.lit(1))
)

# sinistralidade: indenização / prêmio (razão de perda)
df = df.withColumn(
  'sinistralidade',
  F.when(F.col('premio') == 0, F.lit(0.0))
   .otherwise(F.col('indenizacao') / F.col('premio'))
)

# COMMAND ----------

# DBTITLE 1,Filtro de linhas inválidas e reordenação de colunas
# Remove linhas onde UF é inválida (nula ou '0')
df = df.filter((F.col('uf') != '0') & (F.col('uf').isNotNull()))

# Inclui apenas colunas que existem no DataFrame
colunas_finais = [c for c in COLUNAS_FINAIS if c in df.columns]
df = df.select(colunas_finais)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correção de Municípios

# COMMAND ----------

# DBTITLE 1,Função simplificar_nomes (PySpark)
def simplificar_nomes_spark(df, col_name):
  """
  Equivalente PySpark da função simplificar_nomes do notebook original.
  Aplica: lowercase, trim, espaços → '_', remove apóstrofos, hífens → '_',
  remove acentos (transliteration) e substitui 'th' → 't' (São Thomé).
  """
  return (df
    .withColumn(col_name, F.lower(F.trim(F.col(col_name))))
    .withColumn(col_name, F.regexp_replace(F.col(col_name), r'\s+', '_'))
    .withColumn(col_name, F.regexp_replace(F.col(col_name), r"'", ''))
    .withColumn(col_name, F.regexp_replace(F.col(col_name), r'-', '_'))
    .withColumn(col_name, F.translate(F.col(col_name),
                                      'áéíóúâêôãõçàü',
                                      'aeiouaeooocau'))
    .withColumn(col_name, F.regexp_replace(F.col(col_name), 'th', 't'))
  )

# Aplica nos nomes de municípios do DataFrame principal
df = simplificar_nomes_spark(df, 'nome_mun')

# COMMAND ----------

# DBTITLE 1,Correção de nomes de municípios

# Broadcast join para corrigir nomes (dicionário definido em src/lib/const.py)
replacers_pd = pd.DataFrame(list(REPLACERS_MUN.items()), columns=['nome_antigo', 'nome_correto'])
replacers_df = spark.createDataFrame(replacers_pd)

df = (df
  .join(F.broadcast(replacers_df), df['nome_mun'] == replacers_df['nome_antigo'], 'left')
  .withColumn('nome_mun', F.coalesce(F.col('nome_correto'), F.col('nome_mun')))
  .drop('nome_antigo', 'nome_correto')
)

# COMMAND ----------

# DBTITLE 1,Enriquecimento com código IBGE de municípios
def simplificar_nomes_pd(s):
  """Versão pandas da função simplificar_nomes (para normalizar o CSV do IBGE)."""
  s = str(s).strip().lower()
  s = re.sub(r'\s+', '_', s)
  s = s.replace("'", '')
  s = s.replace('-', '_')
  s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
  s = s.replace('th', 't')
  return s

# Lê tabela de municípios do IBGE disponível publicamente
cod_pd = pd.read_csv(IBGE_CSV_URL)
cod_pd['uf'] = cod_pd['uf'].replace(UF_MAP)
cod_pd = cod_pd[['mun', 'nome_mun', 'uf']].copy()
cod_pd['nome_mun'] = cod_pd['nome_mun'].apply(simplificar_nomes_pd)
cod_pd = cod_pd.rename(columns={'mun': 'mun_ibge', 'nome_mun': 'nome_mun_cod'})

cod_spark = spark.createDataFrame(cod_pd).withColumnRenamed('uf', 'uf_ibge')

# Join: preenche código IBGE a partir do nome normalizado + UF
df = (df
  .join(
    F.broadcast(cod_spark),
    (df['nome_mun'] == cod_spark['nome_mun_cod']) & (df['uf'] == cod_spark['uf_ibge']),
    'left'
  )
  .withColumn('mun', F.coalesce(F.col('mun_ibge').cast('int'), F.col('mun').cast('int')))
  .drop('mun_ibge', 'nome_mun_cod', 'uf_ibge')
)

# COMMAND ----------

# DBTITLE 1,Fallback: join com distritos IBGE
# Municípios cujo nome registrado no SISSER corresponde a um distrito — e não ao município-sede.
# O arquivo de distritos é definido em src/lib/const.py (DISTRITOS_PATH).
# Disponível em: https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/divisao_territorial/2022/
try:
  br_pd = pd.read_excel(DISTRITOS_PATH, skiprows=6)
  br_pd = br_pd[['UF', 'Código Município Completo', 'Nome_Município', 'Nome_Distrito']]
  br_pd.columns = ['uf', 'mun', 'nome_mun_sede', 'nome_dist']
  br_pd['uf'] = br_pd['uf'].replace(UF_MAP)
  br_pd['nome_dist'] = br_pd['nome_dist'].apply(simplificar_nomes_pd)
  br_pd['nome_mun_sede'] = br_pd['nome_mun_sede'].apply(simplificar_nomes_pd)
  br_pd = br_pd.rename(columns={'mun': 'mun_dist', 'nome_mun_sede': 'nome_mun_correto'})

  br_spark = spark.createDataFrame(br_pd).withColumnRenamed('uf', 'uf_dist')

  # Registra apenas apólices ainda sem código IBGE após o join principal
  df_com_mun = df.filter(F.col('mun').isNotNull())
  df_sem_mun = df.filter(F.col('mun').isNull()).drop('mun')

  df_sem_mun = (df_sem_mun
    .join(
      F.broadcast(br_spark),
      (df_sem_mun['nome_mun'] == br_spark['nome_dist']) & (df_sem_mun['uf'] == br_spark['uf_dist']),
      'left'
    )
    .withColumn('mun', F.col('mun_dist').cast('int'))
    .withColumn('nome_mun', F.coalesce(F.col('nome_mun_correto'), F.col('nome_mun')))
    .drop('mun_dist', 'nome_mun_correto', 'nome_dist', 'uf_dist')
  )

  df = df_com_mun.unionByName(df_sem_mun)
  print("Fallback de distritos aplicado com sucesso.")
except Exception as e:
  print(f"[AVISO] Arquivo de distritos não encontrado em '{DISTRITOS_PATH}'. Etapa ignorada. Detalhes: {e}")

# COMMAND ----------

# DBTITLE 1,Normalização dos nomes de seguradoras
seg_pd = pd.DataFrame(list(REPLACERS_SEG.items()), columns=['seg_original', 'seg_normalizada'])
seg_spark = spark.createDataFrame(seg_pd)

df = (df
  .join(F.broadcast(seg_spark), df['seguradora'] == seg_spark['seg_original'], 'left')
  .withColumn('seguradora', F.coalesce(F.col('seg_normalizada'), F.col('seguradora')))
  .drop('seg_original', 'seg_normalizada')
)

# COMMAND ----------

# DBTITLE 1,Criação da coluna regiao
regiao_expr = create_map([F.lit(x) for x in chain(*REGIAO_MAP.items())])
df = df.withColumn('regiao', F.coalesce(regiao_expr[F.col('uf')], F.lit('Outro')))

# COMMAND ----------

# DBTITLE 1,Reordenação final das colunas (incluindo regiao)
colunas_finais_v2 = [c for c in COLUNAS_FINAIS if c in df.columns]
df = df.select(colunas_finais_v2)

# Remove linhas onde todos os campos são nulos
df = df.dropna(how='all')

display(df)

# COMMAND ----------

# DBTITLE 1,Escrita na camada Silver
(
  df.write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable(TABLE_SILVER_CLEANED)
)

print(f"Escrita concluída: {df.count()} linhas gravadas em {TABLE_SILVER_CLEANED}")
