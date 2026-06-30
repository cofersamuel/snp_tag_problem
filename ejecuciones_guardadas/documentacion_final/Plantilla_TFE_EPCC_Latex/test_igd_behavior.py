import numpy as np

# Let's write the IGD+ logic manually to see if it matches Pymoo
def calc_igd_plus(pf, F):
    # pf = reference front, F = found front
    # For each z in pf, min over x in F of d+(z, x)
    # d+(z, x) = sqrt(sum_m (max(0, x_m - z_m))^2)
    distances = []
    for z in pf:
        min_d = float('inf')
        for x in F:
            d = np.sqrt(np.sum(np.maximum(0, x - z)**2))
            if d < min_d:
                min_d = d
        distances.append(min_d)
    return np.mean(distances)

ref = np.array([[0.0, 0.0]])
F_bad = np.array([[1.0, 1.0]])
F_good = np.array([[0.0, 0.0]])

print("Manual IGD+ bad:", calc_igd_plus(ref, F_bad))
print("Manual IGD+ good:", calc_igd_plus(ref, F_good))

# What if F_bad is actually [0, 0] and ref is [1, 1]?
print("Manual IGD+ (Found=0, Ref=1):", calc_igd_plus(np.array([[1.0,1.0]]), np.array([[0.0,0.0]])))

