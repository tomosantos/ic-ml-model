# Databricks notebook source
# DBTITLE 1,Setup
# Ingestão e cálculo do SPI (Standardized Precipitation Index) por município
#
# Avaliação de fontes:
#   - INMET API (https://apitempo.inmet.gov.br): cobertura limitada (~700 estações),
#     muitos municípios rurais sem estação próxima → inviável como fonte primária.
#   - ERA5 (ECMWF Reanalysis v5 via cdsapi): cobertura global em grade 0.25°,
#     disponível a partir de 1940, sem lacunas → fonte adotada.
#
# Fluxo:
#   1. Download de precipitação mensal ERA5 (variável: total_precipitation)
#      para o bounding box Brasil via cdsapi.
#   2. Interpolação para centroides dos municípios (tabela IBGE 2022).
#   3. Cálculo do SPI-3 e SPI-6 por município usando scipy.stats.
#   4. Escrita em 00_raw.clima.spi_municipio_mensal.
#
# Pré-requisito: secret scope 'cds' com chave 'cds_api_key' configurada no
#   Databricks. Formato: '<uid>:<key>' (conforme ~/.cdsapirc).

%pip install -q cdsapi netCDF4 scipy numpy pandas xarray

# COMMAND ----------
# DBTITLE 1,Imports
import sys
import os
import math
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType, LongType

sys.path.insert(0, '../lib')
from const import (
    TABLE_RAW_SPI,
    TABLE_SILVER_CLEANED,
    VOLUME_CLIMA,
    DISTRITOS_PATH,
)

# COMMAND ----------
# DBTITLE 1,Parâmetros
dbutils.widgets.text('ano_inicio', '2010')
dbutils.widgets.text('ano_fim',    '2025')

ANO_INICIO = int(dbutils.widgets.get('ano_inicio'))
ANO_FIM    = int(dbutils.widgets.get('ano_fim'))

# Cache local dos arquivos NetCDF no Volume
os.makedirs(VOLUME_CLIMA, exist_ok=True)
ERA5_FILE = f'{VOLUME_CLIMA}/era5_precip_brasil_{ANO_INICIO}_{ANO_FIM}.nc'

print(f"Período: {ANO_INICIO}–{ANO_FIM}")
print(f"Arquivo ERA5: {ERA5_FILE}")

# COMMAND ----------
# DBTITLE 1,Download ERA5 — precipitação mensal Brasil
# Bounding box Brasil: lat [-35, 6], lon [-74, -28]
# total_precipitation em m/mês; convertido para mm/mês nos passos seguintes.

if not Path(ERA5_FILE).exists():
    cds_key = dbutils.secrets.get(scope='cds', key='cds_api_key')
    client = cdsapi.Client(url='https://cds.climate.copernicus.eu/api/v2', key=cds_key)

    years  = [str(y) for y in range(ANO_INICIO, ANO_FIM + 1)]
    months = [f'{m:02d}' for m in range(1, 13)]

    print(f"Baixando ERA5 para {len(years)} anos × 12 meses …")
    client.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable':     'total_precipitation',
            'year':         years,
            'month':        months,
            'time':         '00:00',
            'area':         [6, -74, -35, -28],   # N, W, S, E
            'format':       'netcdf',
        },
        ERA5_FILE,
    )
    print("✓ Download ERA5 concluído")
else:
    print(f"ℹ Usando cache: {ERA5_FILE}")

# COMMAND ----------
# DBTITLE 1,Carrega centroides de municípios IBGE
# Usa a tabela Silver para obter a lista de municípios com lat/lon
df_mun_spark = spark.table(TABLE_SILVER_CLEANED) \
    .select('mun', 'lat', 'lon') \
    .dropDuplicates(['mun']) \
    .filter(F.col('lat').isNotNull() & F.col('lon').isNotNull())

df_mun = df_mun_spark.toPandas()
df_mun['mun'] = df_mun['mun'].astype(str)
print(f"Municípios com coordenadas: {len(df_mun):,}")

# COMMAND ----------
# DBTITLE 1,Interpolação ERA5 → centroide de cada município
ds = xr.open_dataset(ERA5_FILE)

# ERA5 Land usa 'tp' (total_precipitation em m); converte para mm
precip = ds['tp'] * 1000   # mm/mês

records = []
for _, row in df_mun.iterrows():
    lat, lon = float(row['lat']), float(row['lon'])
    mun = row['mun']

    # Seleciona o ponto de grade mais próximo
    serie = precip.sel(latitude=lat, longitude=lon, method='nearest')

    for ts in serie.time.values:
        t = pd.Timestamp(ts)
        val = float(serie.sel(time=ts).values)
        records.append({'mun': mun, 'ano': t.year, 'mes': t.month, 'precip_mm': val})

df_precip = pd.DataFrame(records)
print(f"Registros de precipitação interpolada: {len(df_precip):,}")

# COMMAND ----------
# DBTITLE 1,Cálculo do SPI por município
def calc_spi(series: np.ndarray, scale: int) -> np.ndarray:
    """Calcula o SPI para uma série mensal usando distribuição Gamma (SciPy).

    Parameters
    ----------
    series : array de precipitação mensal em mm
    scale  : janela de acumulação (3 → SPI-3, 6 → SPI-6)

    Returns
    -------
    Array do mesmo comprimento com valores SPI (NaN nos primeiros `scale-1` índices).
    """
    n = len(series)
    spi = np.full(n, np.nan, dtype=float)

    # Acumulado de `scale` meses
    accum = np.convolve(series, np.ones(scale, dtype=float), mode='full')[:n]
    accum[:scale - 1] = np.nan

    valid = ~np.isnan(accum) & (accum > 0)
    if valid.sum() < 20:
        return spi   # Dados insuficientes para ajustar a distribuição

    # Ajuste Gamma no subconjunto válido
    a, loc, scale_param = stats.gamma.fit(accum[valid], floc=0)
    cdf = stats.gamma.cdf(accum, a, loc=loc, scale=scale_param)
    # CDF→ distribuição normal padrão
    spi[valid] = stats.norm.ppf(np.clip(cdf[valid], 1e-6, 1 - 1e-6))
    return spi


spi_rows = []

for mun, grp in df_precip.groupby('mun'):
    grp = grp.sort_values(['ano', 'mes']).reset_index(drop=True)
    precip_arr = grp['precip_mm'].values

    spi3 = calc_spi(precip_arr, 3)
    spi6 = calc_spi(precip_arr, 6)

    for i, (_, row) in enumerate(grp.iterrows()):
        spi_rows.append({
            'mun':   mun,
            'ano':   int(row['ano']),
            'mes':   int(row['mes']),
            'spi_3m': round(float(spi3[i]), 4) if not math.isnan(spi3[i]) else None,
            'spi_6m': round(float(spi6[i]), 4) if not math.isnan(spi6[i]) else None,
        })

df_spi = pd.DataFrame(spi_rows)
print(f"Registros SPI gerados: {len(df_spi):,}")

# COMMAND ----------
# DBTITLE 1,Escrita na tabela Delta
df_spark_spi = spark.createDataFrame(df_spi) \
    .withColumn('mun',   F.col('mun').cast(StringType())) \
    .withColumn('ano',   F.col('ano').cast(IntegerType())) \
    .withColumn('mes',   F.col('mes').cast(IntegerType())) \
    .withColumn('spi_3m', F.col('spi_3m').cast(DoubleType())) \
    .withColumn('spi_6m', F.col('spi_6m').cast(DoubleType()))

spark.sql("CREATE SCHEMA IF NOT EXISTS 01_bronze.seg_rural")

df_spark_spi.write \
    .format('delta') \
    .mode('overwrite') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(TABLE_RAW_SPI)

print(f"✓ {TABLE_RAW_SPI} escrita com {df_spark_spi.count():,} linhas")

# COMMAND ----------
# DBTITLE 1,Verificação
display(spark.table(TABLE_RAW_SPI).limit(100))
