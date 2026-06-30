import matplotlib.pyplot as plt
import numpy as np

# Configurar estilo para que parezca académico
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

fig, ax = plt.subplots(figsize=(8, 6))

# Datos Alta Diversidad (Range Alto)
x_high = [1.0, 2.5, 4.5, 6.0]
y_high = [6.0, 3.5, 1.8, 1.0]

# Datos Baja Diversidad (Range Bajo)
x_low = [3.5, 3.8, 4.2]
y_low = [4.5, 4.0, 3.8]

# Plot de puntos
ax.scatter(x_high, y_high, c='royalblue', marker='o', s=120, label='Alta diversidad (Range Alto)', alpha=0.9, edgecolors='black')
ax.scatter(x_low, y_low, c='crimson', marker='s', s=120, label='Baja diversidad (Range Bajo)', alpha=0.9, edgecolors='black')

# Proyecciones a ejes para Alta Diversidad (azules)
ax.vlines(x=min(x_high), ymin=0, ymax=max(y_high), colors='royalblue', linestyles='dashed', alpha=0.6)
ax.vlines(x=max(x_high), ymin=0, ymax=min(y_high), colors='royalblue', linestyles='dashed', alpha=0.6)
ax.hlines(y=min(y_high), xmin=0, xmax=max(x_high), colors='royalblue', linestyles='dashed', alpha=0.6)
ax.hlines(y=max(y_high), xmin=0, xmax=min(x_high), colors='royalblue', linestyles='dashed', alpha=0.6)

# Proyecciones a ejes para Baja Diversidad (rojos)
ax.vlines(x=min(x_low), ymin=0, ymax=max(y_low), colors='crimson', linestyles='dashed', alpha=0.6)
ax.vlines(x=max(x_low), ymin=0, ymax=min(y_low), colors='crimson', linestyles='dashed', alpha=0.6)
ax.hlines(y=min(y_low), xmin=0, xmax=max(x_low), colors='crimson', linestyles='dashed', alpha=0.6)
ax.hlines(y=max(y_low), xmin=0, xmax=min(x_low), colors='crimson', linestyles='dashed', alpha=0.6)

# Flechas indicadoras de rango
# Rango X alto
ax.annotate('', xy=(min(x_high), 0.3), xytext=(max(x_high), 0.3),
            arrowprops=dict(arrowstyle='<->', color='royalblue', lw=2.5))
ax.text((min(x_high)+max(x_high))/2, 0.45, 'Rango $f_1$', color='royalblue', ha='center', va='bottom', weight='bold')

# Rango X bajo
ax.annotate('', xy=(min(x_low), 0.8), xytext=(max(x_low), 0.8),
            arrowprops=dict(arrowstyle='<->', color='crimson', lw=2.5))
ax.text((min(x_low)+max(x_low))/2, 0.95, 'Rango $f_1$', color='crimson', ha='center', va='bottom', weight='bold')

# Rango Y alto
ax.annotate('', xy=(0.3, min(y_high)), xytext=(0.3, max(y_high)),
            arrowprops=dict(arrowstyle='<->', color='royalblue', lw=2.5))
ax.text(0.45, (min(y_high)+max(y_high))/2, 'Rango $f_2$', color='royalblue', ha='left', va='center', weight='bold', rotation=90)

# Rango Y bajo
ax.annotate('', xy=(0.8, min(y_low)), xytext=(0.8, max(y_low)),
            arrowprops=dict(arrowstyle='<->', color='crimson', lw=2.5))
ax.text(0.95, (min(y_low)+max(y_low))/2, 'Rango $f_2$', color='crimson', ha='left', va='center', weight='bold', rotation=90)


# Limites y etiquetas
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)
ax.set_xlabel('Objetivo $f_1$', fontweight='bold')
ax.set_ylabel('Objetivo $f_2$', fontweight='bold')
ax.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)

plt.tight_layout()
plt.savefig('/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/documentacion_final/figuras_metricas/range_metric.pdf', format='pdf', bbox_inches='tight')
plt.savefig('/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/documentacion_final/figuras_metricas/range_metric.png', format='png', dpi=300, bbox_inches='tight')
