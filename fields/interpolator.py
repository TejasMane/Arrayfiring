import arrayfire as af
import numpy as np
# WORKING ZONE FINDER (VECTORIZED)

def zone_finder(x, y, x_grid, y_grid, Lx, Ly, ghost_cells):

  nx = (x_grid.elements() - 1 - 2 * ghost_cells)  # number of zones in physical domain
  dx = Lx/nx
 
  ny = (y_grid.elements() - 1 - 2 * ghost_cells )  # number of zones in physical domain
  dy = Ly/ny

  x_zone = (((af.abs(x - af.sum(x_grid[0])))/(dx)).as_type(af.Dtype.u32))
  y_zone = (((af.abs(y - af.sum(y_grid[0])))/(dy)).as_type(af.Dtype.u32))

  x_frac = (x - x_grid[x_zone])/dx
  y_frac = (y - y_grid[y_zone])/dy

  af.eval(x_zone, y_zone)
  af.eval(x_frac, y_frac)

  return x_zone, y_zone, x_frac, y_frac


def fraction_finder(x, y, x_grid, y_grid):

  n_x = x_grid.elements()
  n_y = y_grid.elements()

  x_frac = (x - af.sum(x_grid[0])) / af.sum(x_grid[n_x - 1]- x_grid[0])
  y_frac = (y - af.sum(y_grid[0])) / af.sum(y_grid[n_y - 1]- x_grid[0])

  x_frac = x_frac * (n_x-1)
  y_frac = y_frac * (n_y-1)

  af.eval(x_frac, y_frac)

  return x_frac, y_frac

#print('TESTING INTERPOLATOR')

#Lx = Ly =1

#x_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#y_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#x_grid = af.to_array(x_grid)
#y_grid = af.to_array(y_grid)


#x = af.to_array(np.array([0.2, 0.9, 1.2]))
#y = af.to_array(np.array([-0.2, 0.1, 0.4]))

#a1, a2 ,a3, a4 = zone_finder(x, y, x_grid, y_grid, 1, 1, 1)

#print('a1 is ',a1)
#print('a2 is ',a2)
#print('a3 is ',a3)
#print('a4 is ',a4)


#data = np.array([[1.0, 1.0],[3.0, 3.0]])
#data = af.to_array(data)
##data2 = np.array([[0.0, 0.0],[-5.0, -5.0]])
##data2 = af.to_array(data2)

##data3 = af.join(2,data,data2)
##print(data3)
#ans = af.signal.approx2(data, a2[:,1], a2[:,0])
#print(ans)
##print(ans[:,:,1])
