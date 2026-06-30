import matplotlib.pyplot as plt
import numpy as np

def create_tolerance_rate_example():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # 1. Setup Data: Pareto Front
    snps = np.array([10, 25, 50, 100, 200, 400])
    tolerance = np.sqrt(snps) * 1.5

    # 2. Calculate Tolerance Rates (f2 / f1)
    rates = tolerance / snps
    max_rate = np.max(rates)
    avg_rate = np.mean(rates)

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Pareto Front
    ax1.plot(snps, tolerance, 'o-', color='#2C3E50', markersize=10, linewidth=2.5, label='Frente de Pareto')
    for i, txt in enumerate(rates):
        if i >= len(rates) - 2:
            ax1.annotate(f'TR={txt:.2f}', (snps[i], tolerance[i]), textcoords="offset points", xytext=(-10,0), ha='right', va='center', fontsize=11, fontweight='bold')
        else:
            ax1.annotate(f'TR={txt:.2f}', (snps[i], tolerance[i]), textcoords="offset points", xytext=(10,0), ha='left', va='center', fontsize=11, fontweight='bold')

    ax1.set_xlabel('Objetivo $f_1$: Compacidad', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Objetivo $f_2$: Tolerancia', fontsize=14, fontweight='bold')
    ax1.set_title('Frente de Pareto: Tolerancia vs Compacidad', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', frameon=True, shadow=True)

    # Plot 2: Tolerance Rate Evolution
    ax2.bar(range(len(snps)), rates, color='#27AE60', alpha=0.8, edgecolor='black', label='Tasa de Tolerancia (TR)')
    ax2.axhline(y=max_rate, color='#E74C3C', linestyle='--', linewidth=3, label=f'MaxToleranceRate ({max_rate:.2f})')
    ax2.axhline(y=avg_rate, color='#3498DB', linestyle='--', linewidth=3, label=f'AvgToleranceRate ({avg_rate:.2f})')

    ax2.set_xticks(range(len(snps)))
    ax2.set_xticklabels([f'{s}' for s in snps], fontsize=12)
    ax2.set_ylabel('Tasa de Tolerancia ($f_2 / f_1$)', fontsize=14, fontweight='bold')
    ax2.set_title('Análisis de eficiencia', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save
    save_path_pdf = '/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/documentacion_final/figuras_metricas/tolerance_rate_metric.pdf'
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    create_tolerance_rate_example()
