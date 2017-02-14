import numpy as np
import arrayfire as af

def zone_finder(x, y, x_grid, y_grid, Lx, Ly, ghost_cells):

  x_zone = af.data.constant(0, x.elements(), dtype=af.Dtype.f64)
  y_zone = af.data.constant(0, y.elements(), dtype=af.Dtype.f64)
  x_frac = af.data.constant(0, x.elements(), dtype=af.Dtype.f64)
  y_frac = af.data.constant(0, x.elements(), dtype=af.Dtype.f64)
  
  
  nx = (x_grid.elements() - 1 - 2 * ghost_cells)  # number of zones
  dx = Lx/nx
  
  ny = (y_grid.elements() - 1 - 2 * ghost_cells)  # number of zones
  dy = Ly/ny
  
  for i in range(x.elements()):
    

    
    x_zone[i] = int(af.sum(nx * af.abs(x[i] - x_grid[0]))/Lx)  # indexing from zero itself
    y_zone[i] = int(af.sum(ny * af.abs(y[i] - y_grid[0]))/Ly)  # indexing from zero itself
    x_frac[i] = (x[i]-x_grid[af.sum(x_zone[i])])/dx
    y_frac[i] = (y[i]-y_grid[af.sum(y_zone[i])])/dy
  
  return af.join(1, x_zone, y_zone),af.join(1, x_frac, y_frac)

# Testing zone finder
#Lx = Ly =1

#x_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#y_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#x_grid = af.to_array(x_grid)
#y_grid = af.to_array(y_grid)


#x = af.to_array(np.array([0.2, 0.9, 1.2]))
#y = af.to_array(np.array([-0.2, 0.1, 0.4]))

#a1, a2 = zone_finder(x, y, x_grid, y_grid, 1, 1, 1)

##print('a1 is ',a1)
##print('a2 is ',a2)

#data = np.array([[1.0, 1.0],[3.0, 3.0]])
#data = af.to_array(data)
#data2 = np.array([[0.0, 0.0],[-5.0, -5.0]])
#data2 = af.to_array(data2)

#data3 = af.join(2,data,data2)
##print(data3)
#ans = af.signal.approx2(data3, a2[:,1], a2[:,0])
#print(ans)
#print(ans[:,:,1])
