import numpy as np
from pymoo.indicators.igd_plus import IGDPlus

# Reference front (Good points near 0)
ref = np.array([[0.1, 0.1]])

# Bad front (Points near 1)
bad_front = np.array([[0.9, 0.9]])

# Good front (Points near 0)
good_front = np.array([[0.1, 0.1]])

igd = IGDPlus(ref)

print("IGD+ for bad front:", igd(bad_front))
print("IGD+ for good front:", igd(good_front))
