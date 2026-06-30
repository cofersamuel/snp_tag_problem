import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

np.random.seed(123)

# Generar puntos idénticos para todos los escenarios: una nube con distribución normal
# Esto hará que los bordes sean irregulares de manera natural (como en un MOEA real)
n_inside = 250
angles = np.random.uniform(0, 2*np.pi, n_inside)
radii = np.random.normal(1.8, 1.2, n_inside)
radii = np.abs(radii) # Evitar radios negativos

# Puntos adicionales en la periferia para estirar el frente hacia los extremos
n_extra = 50
angles_extra = np.random.uniform(0, 2*np.pi, n_extra)
radii_extra = np.random.uniform(2.5, 3.8, n_extra)

angles = np.concatenate((angles, angles_extra))
radii = np.concatenate((radii, radii_extra))

x = 5 + radii * np.cos(angles)
y = 5 + radii * np.sin(angles)
points = np.column_stack((x, y))
points = np.clip(points, 1.0, 9.0)

def get_pareto_front(pts, opt_f1='min', opt_f2='min'):
    is_pareto = np.zeros(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        dominated = False
        for j in range(pts.shape[0]):
            if i == j: continue
            
            if opt_f1 == 'min':
                dom_f1 = pts[j,0] <= pts[i,0]
                strict_f1 = pts[j,0] < pts[i,0]
            else:
                dom_f1 = pts[j,0] >= pts[i,0]
                strict_f1 = pts[j,0] > pts[i,0]
                
            if opt_f2 == 'min':
                dom_f2 = pts[j,1] <= pts[i,1]
                strict_f2 = pts[j,1] < pts[i,1]
            else:
                dom_f2 = pts[j,1] >= pts[i,1]
                strict_f2 = pts[j,1] > pts[i,1]
                
            if dom_f1 and dom_f2 and (strict_f1 or strict_f2):
                dominated = True
                break
        if not dominated:
            is_pareto[i] = True
            
    return is_pareto

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

scenarios = [
    (axs[0,0], 'min', 'min', 'Minimizar $f_1$ - Minimizar $f_2$'),
    (axs[0,1], 'max', 'min', 'Maximizar $f_1$ - Minimizar $f_2$'),
    (axs[1,0], 'min', 'max', 'Minimizar $f_1$ - Maximizar $f_2$'),
    (axs[1,1], 'max', 'max', 'Maximizar $f_1$ - Maximizar $f_2$')
]

for ax, o1, o2, title in scenarios:
    is_pareto = get_pareto_front(points, o1, o2)
    
    pf_points = points[is_pareto]
    idx = np.argsort(pf_points[:, 0])
    pf_points = pf_points[idx]
    
    ax.scatter(points[~is_pareto, 0], points[~is_pareto, 1], color='lightgray', label='Soluciones dominadas', alpha=0.7)
    ax.scatter(pf_points[:, 0], pf_points[:, 1], color='crimson', s=60, label='Frente de Pareto', zorder=5, edgecolors='black')
    ax.plot(pf_points[:, 0], pf_points[:, 1], color='crimson', linewidth=2, zorder=4)
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Objetivo $f_1$')
    ax.set_ylabel('Objetivo $f_2$')
    
    x_ideal = 1.0 if o1 == 'min' else 9.0
    y_ideal = 1.0 if o2 == 'min' else 9.0
    
    # Flecha del ideal
    ax.annotate('', xy=(x_ideal, y_ideal), xytext=(5, 5),
                arrowprops=dict(facecolor='black', shrink=0.1, width=2, headwidth=8, alpha=0.15),
                zorder=1)
                
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

axs[0,0].legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/documentacion_final/figuras_metricas/pareto_fronts.pdf', format='pdf', bbox_inches='tight')
