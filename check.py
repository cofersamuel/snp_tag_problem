import numpy as np
import os
H = np.loadtxt('data/hapmap_phase2/genotypes_chr21_CEU_r21_nr_fwd_legend.txt', usecols=[0]) # this is not haplotype matrix
import pandas as pd
df = pd.read_csv('resultados/medium/0_datos_previos/matriz_haplotipos_seleccionada_medium.csv', index_col=0)
H = df.values
n_hap, n_snp = H.shape
print(f"Haplotypes: {n_hap}, SNPs: {n_snp}")

diffs = []
for i in range(n_hap):
    for j in range(i+1, n_hap):
        d = np.sum(H[i] != H[j])
        diffs.append(d)
        
diffs = np.array(diffs)
print(f"Min diff: {np.min(diffs)}, Max diff: {np.max(diffs)}, Mean: {np.mean(diffs)}")
