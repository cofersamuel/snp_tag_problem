import pandas as pd
import numpy as np

# Load the detailed results for Experiment B
df = pd.read_csv('/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/30_04_2026/experimentos/full/hinds2005/greedy_hybrid-greedy_ting-random_dense-random_sparse/20260430T183144/1_ejecuciones/resultados_detallados_full.csv')

# Group by algorithm and initialization and calculate means
metrics = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
means = df.groupby(['algorithm', 'init'])[metrics].mean().reset_index()

# Calculate Ranks (1 is best)
# RG (+): High is better -> ascending=False
# SM (-): Low is better -> ascending=True
# MS (-): Low is better -> ascending=True
# MT (+): High is better -> ascending=False
# AT (+): High is better -> ascending=False
# AH (+): High is better -> ascending=False

ranks = pd.DataFrame()
ranks['Method'] = means['algorithm'] + " (" + means['init'] + ")"
ranks['RG'] = means['Range'].rank(ascending=False)
ranks['SM'] = means['SumMin'].rank(ascending=True)
ranks['MS'] = means['MinSum'].rank(ascending=True)
ranks['MT'] = means['MaxToleranceRate'].rank(ascending=False)
ranks['AT'] = means['AvgToleranceRate'].rank(ascending=False)
ranks['AH'] = means['AvgHammingDistance'].rank(ascending=False)

# Calculate Average Rank
ranks['Avg Rank'] = ranks[['RG', 'SM', 'MS', 'MT', 'AT', 'AH']].mean(axis=1)

# Sort by Average Rank
ranks = ranks.sort_values('Avg Rank').reset_index(drop=True)

print(ranks.to_markdown())
