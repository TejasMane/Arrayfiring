import numpy as np
import numpy.linalg as la


"""------------------------Basic-Bilinear-Interpolation-Function(Scipy libraries can also be used)------------------ """
"""
check under alternative algorithm on
https://en.wikipedia.org/wiki/Bilinear_interpolation
for more details regarding the algorithm used.
"""

# returns the interpolated field at x,y position on the particles spatial grid with the field's grid systems
# ( x_grid and y_grid), field values at these grid locations and number of ghost cells provided to the function
def bilinear_interpolate(x, y, x_grid, y_grid, F, ghost_cells, Lx, Ly):
  # x, y are the coordinates at which Interpolated fields have to found
  # x_grid and y_grid are the spatial grids for the field F used
  # F is the electric or magnetic field

  # Storing the field F in temperory variable

  F_function = F
  n = (len(F_function[0, :]) - 1 - 2 * ghost_cells)  # number of zones

  x_zone = int(n * np.float(x - x_grid[0])/Lx)  # indexing from zero itself
  y_zone = int(n * np.float(y - y_grid[0])/Ly)

  # the 4*4 matrix for solving as mention in the wiki page

  A = np.matrix(\
                  [ [1, x_grid[x_zone], y_grid[y_zone], x_grid[x_zone] * y_grid[y_zone]                ], \
                    [1, x_grid[x_zone], y_grid[y_zone + 1], x_grid[x_zone] * y_grid[y_zone + 1]        ], \
                    [1, x_grid[x_zone + 1], y_grid[y_zone], x_grid[x_zone + 1] * y_grid[y_zone]        ], \
                    [1, x_grid[x_zone + 1], y_grid[y_zone + 1], x_grid[x_zone + 1] * y_grid[y_zone + 1]] \
                  ]\
               )

  # The 1D matrix for the points for interpolation

  point_to_calculated_for = np.matrix([[1], [x], [y], [x * y]])

  # Calculating the coefficient matrix

  b = (la.inv(A)).transpose() * point_to_calculated_for

  # assigning the values at the corner points at the grid cell

  Q11 = F_function[y_zone, x_zone]
  Q21 = F_function[y_zone, x_zone + 1]
  Q12 = F_function[y_zone + 1, x_zone]
  Q22 = F_function[y_zone + 1, x_zone + 1]

  Q = np.matrix([[Q11], [Q12], [Q21], [Q22]])

  # Calculating the interpolated value

  F_interpolated = np.float( b.transpose() * Q )

  return F_interpolated


# Vectorizing the interpolation function

bilinear_interpolate = np.vectorize(bilinear_interpolate, excluded=(['x_grid', 'y_grid', 'F','ghost_cells']))


"""-------------------------------------------------------END--------------------------------------------------------"""
