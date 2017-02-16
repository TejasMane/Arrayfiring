import arrayfire as af
#import time as timer

# WORKING ZONE FINDER (POOR VECTORIZATION TIME between 2 and 3)

def zone_finder(x, y, x_grid, y_grid, Lx, Ly, ghost_cells):
  #s = timer.time()
  x_zone = af.data.constant(0, x.elements(), dtype=af.Dtype.u32)
  y_zone = af.data.constant(0, y.elements(), dtype=af.Dtype.u32)
  x_frac = af.data.constant(0, x.elements(), dtype=af.Dtype.u32)
  y_frac = af.data.constant(0, x.elements(), dtype=af.Dtype.u32)

  nx = (x_grid.elements() - 1 - 2 * ghost_cells)  # number of zones
  dx = Lx/nx

  ny = (y_grid.elements() - 1 - 2 * ghost_cells)  # number of zones
  dy = Ly/ny
  #print('1', timer.time()-s)
  x_zone = (((nx * af.abs(x - af.sum(x_grid[0])))/Lx).as_type(af.Dtype.u32))
  y_zone = (((ny * af.abs(y - af.sum(y_grid[0])))/Ly).as_type(af.Dtype.u32))
  #print('2', timer.time()-s)
  x_frac = (x - x_grid[x_zone])/dx
  y_frac = (y - y_grid[y_zone])/dy
  #print('3', timer.time()-s)

  #af.eval(x_zone, y_zone)
  #af.eval(x_frac, y_frac)
  return x_zone, y_zone, x_frac, y_frac


#def zone_finder(x, y, x_grid, y_grid, Lx, Ly, ghost_cells):
  #s = timer.time()
  #x_zone = af.data.constant(0, x.elements(), dtype=af.Dtype.u32)
  #y_zone = af.data.constant(0, y.elements(), dtype=af.Dtype.u32)
  #x_frac = af.data.constant(0, x.elements(), dtype=af.Dtype.u32)
  #y_frac = af.data.constant(0, x.elements(), dtype=af.Dtype.u32)

  #nx = (x_grid.elements() - 1 - 2 * ghost_cells)  # number of zones
  #dx = Lx/nx

  #ny = (y_grid.elements() - 1 - 2 * ghost_cells)  # number of zones
  #dy = Ly/ny
  #print('1', timer.time()-s)
  #x_zone = (((nx * af.abs(x - af.sum(x_grid[0])))/Lx).as_type(af.Dtype.u32))
  #y_zone = (((ny * af.abs(y - af.sum(y_grid[0])))/Ly).as_type(af.Dtype.u32))
  #print('2', timer.time()-s)
  #temp1 = x_grid[x_zone]
  #temp2 = y_grid[y_zone]
  #x_frac = (x - temp1)/dx
  #y_frac = (y - temp2)/dy
  #print('3', timer.time()-s)
  #return af.join(1, x_zone, y_zone),af.join(1, x_frac, y_frac)



#print('RESULTS FOR SECOND INTERPOLATOR')

#Lx = Ly =1

#x_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#y_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#x_grid = af.to_array(x_grid)
#y_grid = af.to_array(y_grid)


#x = af.to_array(np.array([0.2, 0.9, 1.2]))
#y = af.to_array(np.array([-0.2, 0.1, 0.4]))

#a1, a2 = zone_finder2(x, y, x_grid, y_grid, 1, 1, 1)

#print('a1 is ',a1)
#print('a2 is ',a2)

#data = np.array([[1.0, 1.0],[3.0, 3.0]])
#data = af.to_array(data)
##data2 = np.array([[0.0, 0.0],[-5.0, -5.0]])
##data2 = af.to_array(data2)

##data3 = af.join(2,data,data2)
##print(data3)
#ans = af.signal.approx2(data, a2[:,1], a2[:,0])
#print(ans)
##print(ans[:,:,1])
