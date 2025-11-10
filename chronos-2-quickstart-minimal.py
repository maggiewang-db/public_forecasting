# Databricks notebook source
# MAGIC %pip install 'chronos-forecasting>=2.0' 'pandas[pyarrow]' 'matplotlib'

# COMMAND ----------

import os

# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline

# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Univariate Forecasting
# MAGIC
# MAGIC We start with a simple univariate forecasting example using the pandas API.

# COMMAND ----------

# Load data as a long-format pandas data frame
context_df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")
print("Input dataframe shape:", context_df.shape)

print("Preview of input dataframe:")
print(context_df.head().to_string(index=False))

# COMMAND ----------

pred_df = pipeline.predict_df(context_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])

print("Output dataframe shape:", pred_df.shape)

print("Preview of prediction dataframe:")
print(pred_df.head().to_string(index=False))