# # Test script for b1 depositor
import numpy as np
import h5py
import params
import arrayfire as af
import time as timer


from fields.current_depositor import current_b1_depositor
charge = 1
x1 = af.Array([0.2,0.7])
y1 = af.Array([0.4,0.8])
velocity_required = af.Array([1.0,1.0])
x_grid = af.Array([-1.0, 0.0, 1.0, 2.0])
y_grid = af.Array([-1.0, 0.0, 1.0, 2.0])
ghost_cells = 1
Lx = 1.0
Ly = 1.0

print(current_b1_depositor(charge, x1, y1, velocity_required, x_grid, y_grid, ghost_cells, Lx, Ly))
