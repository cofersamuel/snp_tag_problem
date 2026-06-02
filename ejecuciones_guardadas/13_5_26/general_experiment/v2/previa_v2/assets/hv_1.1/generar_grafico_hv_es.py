import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Definir un frente de Pareto simple (minimización)
points = np.array([[0.2, 1.0], [0.5, 0.5], [1.0, 0.2]])
# Ordenar los puntos según el eje X para graficar correctamente
points = points[np.argsort(points[:, 0])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Función para dibujar el área dominada (hipervolumen)
def draw_hypervolume(ax, pts, ref_point, title):
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=12, pad=15, fontweight='bold')
    ax.set_xlabel('Objetivo 1', fontsize=12)
    ax.set_ylabel('Objetivo 2', fontsize=12)
    
    # Graficar punto de referencia
    ax.plot(ref_point[0], ref_point[1], 'ro', markersize=10, label=f'Referencia {ref_point}')
    
    # Graficar puntos de las soluciones
    ax.plot(pts[:, 0], pts[:, 1], 'ko', markersize=8, label='Soluciones de Pareto')
    
    # Anotar los puntos
    ax.annotate('A (0.2, 1.0)', (pts[0,0]-0.05, pts[0,1]+0.03), fontsize=10, fontweight='bold')
    ax.annotate('B (0.5, 0.5)', (pts[1,0]+0.03, pts[1,1]+0.03), fontsize=10, fontweight='bold')
    ax.annotate('C (1.0, 0.2)', (pts[2,0]+0.03, pts[2,1]-0.05), fontsize=10, fontweight='bold')
    
    # Dibujar las áreas sombreadas (unión de rectángulos)
    for p in pts:
        width = ref_point[0] - p[0]
        height = ref_point[1] - p[1]
        if width > 0 and height > 0:
            rect = patches.Rectangle((p[0], p[1]), width, height, linewidth=1.5, edgecolor='blue', facecolor='lightblue', alpha=0.35)
            ax.add_patch(rect)
            
    # Dibujar líneas discontinuas que marcan los límites del punto de referencia
    ax.axhline(y=ref_point[1], color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=ref_point[0], color='r', linestyle='--', alpha=0.5)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=10)

# Gráfico 1: Punto de referencia [1.0, 1.0]
draw_hypervolume(ax1, points, [1.0, 1.0], 'Hipervolumen con Punto de Referencia [1.0, 1.0]\n(Las soluciones extremas A y C aportan volumen 0)')

# Gráfico 2: Punto de referencia [1.1, 1.1]
draw_hypervolume(ax2, points, [1.1, 1.1], 'Hipervolumen con Punto de Referencia [1.1, 1.1]\n(El "buffer" de 0.1 permite que A y C aporten volumen positivo)')

plt.tight_layout()
# Guardar la imagen en el mismo directorio de hv_1.1
plt.savefig('/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/13_5_26/general_experiment/v2/hv_1.1/hypervolume_comparison_es.png', dpi=150)
print("Gráfico generado y guardado con éxito como hypervolume_comparison_es.png.")
