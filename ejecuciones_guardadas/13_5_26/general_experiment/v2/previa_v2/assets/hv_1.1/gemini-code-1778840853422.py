import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Define a simple Pareto front (minimization)
points = np.array([[0.2, 1.0], [0.5, 0.5], [1.0, 0.2]])
# Sort points by x-axis for step plotting
points = points[np.argsort(points[:, 0])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Function to draw dominated area
def draw_hypervolume(ax, pts, ref_point, title):
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('Objective 1', fontsize=12)
    ax.set_ylabel('Objective 2', fontsize=12)
    
    # Plot reference point
    ax.plot(ref_point[0], ref_point[1], 'ro', markersize=10, label=f'Reference {ref_point}')
    
    # Plot points
    ax.plot(pts[:, 0], pts[:, 1], 'ko', markersize=8, label='Pareto Solutions')
    
    # Annotate points
    ax.annotate('A (0.2, 1.0)', (pts[0,0]-0.05, pts[0,1]+0.03))
    ax.annotate('B (0.5, 0.5)', (pts[1,0]+0.03, pts[1,1]+0.03))
    ax.annotate('C (1.0, 0.2)', (pts[2,0]+0.03, pts[2,1]-0.05))
    
    # Draw shaded area (union of rectangles)
    for p in pts:
        width = ref_point[0] - p[0]
        height = ref_point[1] - p[1]
        if width > 0 and height > 0:
            rect = patches.Rectangle((p[0], p[1]), width, height, linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.4)
            ax.add_patch(rect)
            
    # Draw dashed lines to show bounds
    ax.axhline(y=ref_point[1], color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=ref_point[0], color='r', linestyle='--', alpha=0.5)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left')

# Plot 1: Reference point [1.0, 1.0]
draw_hypervolume(ax1, points, [1.0, 1.0], 'Hypervolume with Reference Point [1.0, 1.0]\n(Extreme solutions A and C contribute 0 volume)')

# Plot 2: Reference point [1.1, 1.1]
draw_hypervolume(ax2, points, [1.1, 1.1], 'Hypervolume with Reference Point [1.1, 1.1]\n(Buffer allows A and C to contribute positive volume)')

plt.tight_layout()
plt.savefig('hypervolume_comparison.png', dpi=150)
plt.show()