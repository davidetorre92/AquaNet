import sys  
sys.path.append('/home/davide/AI/Projects/AquaNet') # Initialize Experiment

import numpy as np
import pandas as pd
from settings import df_robustness_path, df_eda_path, df_motif_Z_scores_path

# Correlation between robustness and other Food Webs metrics evaluated in the EDA phase.
# Load robustness data
df_robustness = pd.read_pickle(df_robustness_path)
# Load other Food Web metrics
df_eda = pd.read_pickle(df_eda_path)
# Load Motif Z-scores
df_motif_Z_scores = pd.read_pickle(df_motif_Z_scores_path)
df_motif_Z_scores = df_motif_Z_scores[['Network', 'S2']]
# Merge
merged_df = pd.merge(df_robustness, df_eda, on='Network', how='right')
merged_df = pd.merge(merged_df, df_motif_Z_scores, on='Network', how='right')
# Change column name: Giant Strongly Connected Component -> Core fraction
merged_df.rename(columns={'Giant Strongly Connected Component': 'Core fraction'}, inplace=True)
# Correlation
corr_df = merged_df.corr(numeric_only=True)
print(corr_df)