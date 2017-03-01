import numpy as np
import h5py
import params
import arrayfire as af
import time as timer
import pylab as pl

pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'



print(af.device.device_info())


fields_enabled   = params.fields_enabled

if(fields_enabled == "true"):
  spread            = params.spread
  ghost_cells       = params.ghost_cells
  speed_of_light    = params.speed_of_light


left_boundary    = params.left_boundary
right_boundary   = params.right_boundary
length_box_x     = params.length_box_x

bottom_boundary  = params.bottom_boundary
top_boundary     = params.top_boundary
length_box_y     = params.length_box_y

back_boundary    = params.back_boundary
front_boundary   = params.front_boundary
length_box_z     = params.length_box_z

# field_error_convergence(a, b) returns the errors in field variables calculated after the waves come 
# back to their initial position on a spatial grid with a*b dimensions

no_of_particles = 1

def gauss1D(x):

  return af.arith.exp(-( (x - 0.5)**2 )/(2*spread**2))

def sumsum(a):

  return (af.sum(af.abs(a)))


def field_error_convergence(a, b):

  Ez_error = np.zeros(a.elements(), dtype = np.float)
  Bx_error = np.zeros(a.elements(), dtype = np.float)
  By_error = np.zeros(a.elements(), dtype = np.float)
  Bz_error = np.zeros(a.elements(), dtype = np.float)
  Ex_error = np.zeros(a.elements(), dtype = np.float)
  Ey_error = np.zeros(a.elements(), dtype = np.float)
  
  for outer_index in range(a.elements()):

    """ Number of divisions in the physical domain"""
    
    Nx = af.sum(a[outer_index])
    Ny = af.sum(b[outer_index])
    
    print('Grid Size = ',Nx )
    """ Setting division size and time steps"""
    Lx = right_boundary - left_boundary
    Ly = top_boundary - bottom_boundary


    dx = (Lx / (Nx))
    dy = (Ly / (Ny))

    dt = (dx / (2 * speed_of_light))

    final_time = 2
    time = np.arange(0, final_time, dt)

    """ Setting the spatial physical grids """

    x_center = np.linspace(-ghost_cells*dx, Lx + ghost_cells*dx, Nx + 1 + 2 * ghost_cells, endpoint=True)
    y_center = np.linspace(-ghost_cells*dy, Ly + ghost_cells*dy, Ny + 1 + 2 * ghost_cells, endpoint=True)

    """ Setting the offset spatial grids """

    x_right = np.linspace(-ghost_cells * dx / 2, Lx + (2 * ghost_cells + 1) * dx / 2, Nx + 1 + 2 * ghost_cells,\
                            endpoint=True\
                         )

    y_top = np.linspace(-ghost_cells * dy / 2, Ly + (2 * ghost_cells + 1) * dy / 2, Ny + 1 + 2 * ghost_cells,\
                          endpoint=True\
                       )


    x_center = af.to_array(x_center)
    y_center = af.to_array(y_center)
    x_right = af.to_array(x_right)
    y_top = af.to_array(y_top)



    """ Initializing the Fields """

    Ez = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)
    Bx = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)
    By = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)

    Bz = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)
    Ex = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)
    Ey = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)


    X_center_physical = af.tile(af.reorder(x_center[ghost_cells:-ghost_cells],1),y_center[ghost_cells:-ghost_cells].elements(),1)

    X_right_physical  = af.tile(af.reorder(x_right[ghost_cells:-ghost_cells],1),y_center[ghost_cells:-ghost_cells].elements(),1)

    Y_center_physical = af.tile(y_center[ghost_cells:-ghost_cells], 1, x_center[ghost_cells:-ghost_cells].elements())

    Y_top_physical    = af.tile(y_top[ghost_cells:-ghost_cells], 1, x_center[ghost_cells:-ghost_cells].elements())
    
    
    div_B = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)

    """  Setting initial conditions for the fields in the physical domain """

    # Initialize the fields in the manner desired below using X_center_physical, Y_center_physical, X_right_physical
    # and Y_top_physical :

    # Example for the changing the initialization to a 2 dimensional gaussian wave

    # Ez[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = gauss2D(X_center_physical, Y_center_physical)

    Ez[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = gauss1D(X_center_physical)

    By[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = gauss1D(X_right_physical)

    Bz[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = gauss1D(2*X_right_physical)

    Ey[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = gauss1D(2*X_center_physical)

    # Saving the initial conditions in different variables

    Ez_initial = Ez.copy()
    Bx_initial = Bx.copy()
    By_initial = By.copy()

    Bz_initial = Bz.copy()
    Ex_initial = Ex.copy()
    Ey_initial = Ey.copy()

    forward_row     = af.Array([1, -1, 0])
    forward_column  = af.Array([1, -1, 0])
    backward_row    = af.Array([0, 1, -1])
    backward_column = af.Array([0, 1, -1])
    identity        = af.Array([0, 1, 0] )


    """  Starting the solver """
    from fields.fdtd import fdtd
    for time_index,t0 in enumerate(time):
      # print('grid size = ', Nx, 'time = ', time_index)

      Jx, Jy, Jz = 0, 0, 0
      Ex, Ey, Ez, Bx, By, Bz = fdtd(Ex, Ey, Ez, Bx, By, Bz, speed_of_light, Lx, Ly, ghost_cells, Jx, Jy, Jz, dt, no_of_particles)


      #Divergence Computation


      #div_B[X_index, Y_index] = (Bx[X_index, Y_index + 1]-Bx[X_index,Y_index])/(dx) +  (By[X_index + 1, Y_index]-By[X_index, Y_index])/(dy)
    
      #div_B = signal.convolve2d(Bx, forward_column ,'same') + signal.convolve2d(By, forward_row ,'same')

      # Uncomment the following set of lines to write the data to disk
      # make folders as neccessary
      # these lines will write data from first timestep

      #h5f = h5py.File('Ex/solution_'+str(time_index)+'.h5', 'w')
      #h5f.create_dataset('Ex/solution_dataset_'+str(time_index), data=Ex)
      #h5f.close()
      
      #h5f = h5py.File('Ey/solution_'+str(time_index)+'.h5', 'w')
      #h5f.create_dataset('Ey/solution_dataset_'+str(time_index), data=Ey)
      #h5f.close()
      
      #h5f = h5py.File('Ez/solution_'+str(time_index)+'.h5', 'w')
      #h5f.create_dataset('Ez/solution_dataset_'+str(time_index), data=Ez)
      #h5f.close()
      
      #h5f = h5py.File('Bx/solution_'+str(time_index)+'.h5', 'w')
      #h5f.create_dataset('Bx/solution_dataset_'+str(time_index), data=Bx)
      #h5f.close()
      
      #h5f = h5py.File('By/solution_'+str(time_index)+'.h5', 'w')
      #h5f.create_dataset('By/solution_dataset_'+str(time_index), data=By)
      #h5f.close()
      
      #h5f = h5py.File('Bz/solution_'+str(time_index)+'.h5', 'w')
      #h5f.create_dataset('Bz/solution_dataset_'+str(time_index), data=Bz)
      #h5f.close()
      
      #h5f = h5py.File('div/solution_'+str(time_index)+'.h5', 'w')
      #h5f.create_dataset('div/solution_dataset_'+str(time_index), data=div_B)
      #h5f.close()


      # Computing Numerical error after two box crossing timescales
      
      # For arbitrary initial conditions set time_index == time step # where wave comes back to its initial conditions
      # if it is happening

      if (time_index == time.size-1):
        Ez_error[outer_index] = sumsum(Ez[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] - \
                          Ez_initial[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells]\
                          ) / (Nx * Ny)

        Bx_error[outer_index] = sumsum(Bx[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] - \
                          Bx_initial[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells]\
                          ) / (Nx * Ny)

        By_error[outer_index] = sumsum(By[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] - \
                          By_initial[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells]\
                          ) / (Nx * Ny)

        Bz_error[outer_index] = sumsum(Bz[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] - \
                          Bz_initial[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells]\
                          ) / (Nx * Ny)

        Ex_error[outer_index] = sumsum(Ex[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] - \
                          Ex_initial[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells]\
                          ) / (Nx * Ny)

        Ey_error[outer_index] = sumsum(Ey[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] - \
                          Ey_initial[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells]\
                          ) / (Nx * Ny)
                          
  # print('So this is it', Ez_error, Bx_error, By_error, Bz_error, Ex_error, Ey_error)
  return Ez_error, Bx_error, By_error, Bz_error, Ex_error, Ey_error


"""Vectorizing the function """

""" Test grid size ranges """

# Change here according to need

N = af.Array([32, 64, 128, 256, 512])

# for making movies for 100*100 comment the above statement and uncomment the following line

#N = np.array([100])

ErrorNEz, ErrorNBx, ErrorNBy, ErrorNBz, ErrorNEx, ErrorNEy = field_error_convergence(N, N)

print('function output',ErrorNEz, ErrorNBx, ErrorNBy, ErrorNBz, ErrorNEx, ErrorNEy)
# Optional plotting script down below:
# The timing might not be right for modified time_in_seconds/Lx/Ly
# Check before hand

pl.loglog(N, ErrorNEz, '-o', label='$E_z$ ')
pl.loglog(N, ErrorNBx, '-o', label='$B_x$ ')
pl.loglog(N, ErrorNBy, '-o', label='$B_y$ ')
pl.loglog(N, ErrorNBz, label='$B_z$ ')
pl.loglog(N, ErrorNEx, label='$E_x$ ')
pl.loglog(N, ErrorNEy, label='$E_y$ ')
pl.loglog(N, 1.5 * (N ** -1.999), '--', color='black', lw=2, label=' $O(N^{-2})$ ')
pl.legend().draggable()
pl.title('$\mathrm{Convergence\; plot}$ ')
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{L_1\;norm\;of\;error}$')
pl.show()
pl.clf()
