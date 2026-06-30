import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# --- True Pareto Front (Reference) ---
# Curve: y = 10 / (x + 1)
x_ref = np.linspace(1, 9, 8)
y_ref = 10 / (x_ref * 0.5 + 1)

# --- Discovered Front for GD+ ---
# 1 point close to reference, 2 points very far (bad convergence)
x_gd_disc = [2.0, 5.0, 8.0]
y_gd_disc = [5.5, 6.0, 4.0]

# --- Discovered Front for IGD+ ---
# 3 points perfectly converging, but only covering the top-left area (bad coverage)
x_igd_disc = [2.0, 3.0, 4.0]
y_igd_disc = [10 / (2 * 0.5 + 1) + 0.8, 10 / (3 * 0.5 + 1) + 0.8, 10 / (4 * 0.5 + 1) + 0.8]

# =======================================================
# Panel 1: GD+ (From Discovered TO Reference)
# =======================================================
ax1.plot(x_ref, y_ref, c='gray', marker='*', markersize=10, linestyle='-', alpha=0.5, label='Frente óptimo (Referencia)')
ax1.scatter(x_gd_disc, y_gd_disc, c='crimson', marker='o', s=100, label='Soluciones descubiertas', edgecolors='black', zorder=5)

# Arrows from Discovered to Nearest Reference
# x_gd_disc[0]=2, y_gd_disc[0]=5.5 -> nearest is ref[1] (x=2.14, y=4.8) roughly
for i, (xd, yd) in enumerate(zip(x_gd_disc, y_gd_disc)):
    # Find nearest reference point mathematically
    distances = np.sqrt((x_ref - xd)**2 + (y_ref - yd)**2)
    min_idx = np.argmin(distances)
    xr, yr = x_ref[min_idx], y_ref[min_idx]
    
    # Draw arrow from discovered to reference
    ax1.annotate('', xy=(xr, yr), xytext=(xd, yd),
                 arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5, ls='--'))

# Annotations removed

ax1.set_title('Métrica GD+ (Convergencia)', fontweight='bold', pad=15)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.set_xlabel('Objetivo $f_1$')
ax1.set_ylabel('Objetivo $f_2$')
ax1.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)

# =======================================================
# Panel 2: IGD+ (From Reference TO Discovered)
# =======================================================
ax2.plot(x_ref, y_ref, c='gray', marker='*', markersize=10, linestyle='-', alpha=0.5, label='Frente óptimo (Referencia)')
ax2.scatter(x_igd_disc, y_igd_disc, c='royalblue', marker='o', s=100, label='Soluciones descubiertas', edgecolors='black', zorder=5)

# Arrows from Reference to Nearest Discovered
for i, (xr, yr) in enumerate(zip(x_ref, y_ref)):
    # Find nearest discovered point mathematically
    distances = np.sqrt((np.array(x_igd_disc) - xr)**2 + (np.array(y_igd_disc) - yr)**2)
    min_idx = np.argmin(distances)
    xd, yd = x_igd_disc[min_idx], y_igd_disc[min_idx]
    
    # Draw arrow from reference to discovered
    ax2.annotate('', xy=(xd, yd), xytext=(xr, yr),
                 arrowprops=dict(arrowstyle='->', color='royalblue', lw=1.5, ls='--'))

# Annotations removed

ax2.set_title('Métrica IGD+ (Convergencia y Cobertura)', fontweight='bold', pad=15)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.set_xlabel('Objetivo $f_1$')
ax2.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)

plt.tight_layout()
plt.savefig('/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/documentacion_final/figuras_metricas/gd_igd_metric.pdf', format='pdf', bbox_inches='tight')
plt.savefig('/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/documentacion_final/figuras_metricas/gd_igd_metric.png', format='png', dpi=300, bbox_inches='tight')
