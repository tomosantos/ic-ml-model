# Databricks notebook source
dbutils.fs.ls('/Volumes/00_raw/data/seguro_rural')

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG `01_bronze`;
# MAGIC CREATE SCHEMA IF NOT EXISTS 01_bronze.seg_rural;

# COMMAND ----------

table = dbutils.widgets.get('table')
tableName = dbutils.widgets.get('tableName')

df = spark.read.excel(f'/Volumes/00_raw/data/seguro_rural/{table}/', headerRows=1)

(
    df
    .coalesce(1)
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable(f'01_bronze.seg_rural.{tableName}')
)
