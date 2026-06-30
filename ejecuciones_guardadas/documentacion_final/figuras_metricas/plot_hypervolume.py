import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

def create_hypervolume_example():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # 1. Setup Data: Two Pareto Fronts (normalized space [0,1])
    # Algorithm A (Good: Close to origin, well spread with extreme solutions)
    front_a = np.array([[0.0, 1.0], [0.1, 0.8], [0.3, 0.5], [0.5, 0.3], [0.8, 0.1], [1.0, 0.0]])

    # Algorithm B (Poor: Far from origin, clustered)
    front_b = np.array([[0.6, 0.9], [0.7, 0.8], [0.9, 0.6]])

    ref_point = np.array([1.1, 1.1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

    def plot_hv(ax, front, color, title):
        idx = np.argsort(front[:, 0])
        f = front[idx]
        
        for i in range(len(f)):
            width = (f[i+1][0] - f[i][0]) if i < len(f)-1 else (ref_point[0] - f[i][0])
            height = (ref_point[1] - f[i][1])
            rect = Rectangle((f[i][0], f[i][1]), width, height, color=color, alpha=0.3, label='Área Dominada' if i==0 else "")
            ax.add_patch(rect)
            
        ax.scatter(f[:, 0], f[:, 1], color=color, s=80, edgecolors='black', zorder=5, label='Soluciones Pareto')
        ax.scatter(ref_point[0], ref_point[1], color='black', marker='x', s=100, label='Punto Ref. (1.1, 1.1)')
        
        ax.set_xlim(-0.05, 1.25)
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel('Objetivo 1 (Minimizar)', fontsize=12)
        ax.set_ylabel('Objetivo 2 (Minimizar)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        # Legend moved to lower left to avoid overlap
        ax.legend(loc='lower left')
        
        hv_val = 0
        for i in range(len(f)):
            x_start = f[i][0]
            x_end = f[i+1][0] if i < len(f)-1 else ref_point[0]
            y_span = ref_point[1] - f[i][1]
            hv_val += (x_end - x_start) * y_span
        
        ax.text(0.1, 0.25, f'HV Calculado ≈ {hv_val:.3f}', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plot_hv(ax1, front_a, '#3498DB', 'Algoritmo A: Alto Hipervolumen\n(Buena Convergencia y Diversidad)')
    plot_hv(ax2, front_b, '#E74C3C', 'Algoritmo B: Bajo Hipervolumen\n(Pobre Convergencia y Diversidad)')

    plt.tight_layout()

    save_path = '/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/documentacion_final/figuras_metricas/hypervolume_metric.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    create_hypervolume_example()
