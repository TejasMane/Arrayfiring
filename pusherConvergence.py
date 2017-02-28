import h5py
import numpy as np
from integrators.magnetic_verlet import integrator
import arrayfire as af
from fields.fdtd import fdtd
from scipy.integrate import odeint
from fields.interpolator import fraction_finder
from wall_options.periodic import wall_x, wall_y, wall_z

# """Collision Options"""
#
# if(collision_operator == "montecarlo"):
#   from collision_operators.monte_carlo import collision_operator
#
# # We shall define a collision operator for the potential based model and collisionless models as well,
# # Although integrator takes care of the scattering itself. The operator shall return the values as is
# # This is to avoid condition checking inside the time-loop
#
# if(collision_operator == "potential-based"):
#   from collision_operators.potential import collision_operator
#
# if(collision_operator == "collisionless"):
#   from collision_operators.collisionless import collision_operator
# # field_error_convergence(a, b) returns the errors in field variables calculated after the waves come
# # back to their initial position on a spatial grid with a*b dimensions
#
#
#

#Equations are:
# dBz/dt = - dEy/dx
# dEy/dt = - dBz/dx

# x, E in time n, n+1.............
# v, B in time n +0.5, n + 1.5 .....
# for Boris Algorithm : x(n+1) = x(n) + v(n+0.5)dt
#  v(n+1.5) = v(n + 0.5) + fields(E(n+1), B(avg(n+1.5,n+0.5)))


# For analytical comparision
# for x start from x(n+1) and for vx start from time (n+1)


no_of_particles = 1
right_boundary = 1
left_boundary = 0
top_boundary = 1
bottom_boundary = 0
ghost_cells = 1
speed_of_light = 1




def error(a,b):
  for outer_index in range(len(a)):
    Nx = (a[outer_index])
    Ny = (b[outer_index])

    Lx = right_boundary - left_boundary
    Ly = top_boundary - bottom_boundary
    
    
    dx = Lx/Nx
    dy = Ly/Ny

    final_time = 2
    dt = np.float(dx / (2 * speed_of_light))
    time = np.arange(0, final_time, dt)

    def analytical(Y, t):
      x, y, vx, vy = Y
      dydt = [vx, vy, vy * np.sin(2 * np.pi * (t + dt - x)), (1 - vx) * np.sin(2 * np.pi * (t + dt - x))]
      return dydt

    """ Setting the grids """

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

    X_center_physical = af.tile(af.reorder(x_center[ghost_cells:-ghost_cells],1),y_center[ghost_cells:-ghost_cells].elements(),1)

    X_right_physical  = af.tile(af.reorder(x_right[ghost_cells:-ghost_cells],1),y_center[ghost_cells:-ghost_cells].elements(),1)

    Y_center_physical = af.tile(y_center[ghost_cells:-ghost_cells], 1, x_center[ghost_cells:-ghost_cells].elements())

    Y_top_physical    = af.tile(y_top[ghost_cells:-ghost_cells], 1, x_center[ghost_cells:-ghost_cells].elements())


    """ Initial conditions for positions """
    # At n = 0
    x_initial = np.ones((no_of_particles), dtype = np.float)*Lx/2
    # At n = 0
    y_initial = np.ones((no_of_particles), dtype = np.float)*Ly/2
    # At n = 0
    z_initial = np.ones((no_of_particles), dtype = np.float)*Ly/2
    # At n = 0
    """ Setting velocities according to maxwellian distribution """
    # At n = 0.5
    vel_x_initial = np.zeros(no_of_particles, dtype=np.float)
    vel_y_initial = np.zeros(no_of_particles, dtype=np.float)
    vel_z_initial = np.zeros(no_of_particles, dtype=np.float)

    vel_x_initial[:] = 0.2
    vel_y_initial[:] = 0.2

    x_initial = af.to_array(x_initial)
    y_initial = af.to_array(y_initial)
    z_initial = af.to_array(z_initial)
    vel_x_initial = af.to_array(vel_x_initial)
    vel_y_initial = af.to_array(vel_y_initial)
    vel_z_initial = af.to_array(vel_z_initial)


    """ Combining the initial conditions into one vector"""

    initial_conditions = np.concatenate([x_initial, y_initial,\
                                         z_initial, vel_x_initial,\
                                         vel_y_initial, vel_z_initial], axis=0)

    """ Electric and Magnetic field """

    Ez = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    Bx = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    By = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)

    Bz = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    Ex = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    Ey = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)

    Ez_particle = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    Bx_particle = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    By_particle = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)

    Bz_particle = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    Ex_particle = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    Ey_particle = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
    
    
    """ Writing the spatial grids as a two dimension matrix for vectorization purposes """

    X_center_physical = af.tile(af.reorder(x_center[ghost_cells:-ghost_cells], 1),
                                y_center[ghost_cells:-ghost_cells].elements(), 1)

    X_right_physical = af.tile(af.reorder(x_right[ghost_cells:-ghost_cells], 1),
                               y_center[ghost_cells:-ghost_cells].elements(), 1)

    Y_center_physical = af.tile(y_center[ghost_cells:-ghost_cells], 1, x_center[ghost_cells:-ghost_cells].elements())

    Y_top_physical = af.tile(y_top[ghost_cells:-ghost_cells], 1, x_center[ghost_cells:-ghost_cells].elements())

    """ Discretizing time and making sure scaling is done right """

    # box_crossing_time_scale = length_of_box_x / np.max(initial_conditions_velocity_x)


    # Initializing the non relevant fields:

    Ey[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = af.arith.sin(2*np.pi*(-X_right_physical))
    Bz[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = af.arith.sin(2*np.pi*((dt/2)-X_right_physical))

    #Bz[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = 20



    position_analytical = np.zeros((len(time),2), dtype = np.float)
    velocity_analytical = np.zeros((len(time),2), dtype = np.float)

    position_numerical = np.zeros((len(time),2), dtype = np.float)
    velocity_numerical = np.zeros((len(time),2), dtype = np.float)

    Num_error = np.zeros((len(time),2), dtype = np.float)


    """ Solving """

    old = np.zeros(6 * no_of_particles, dtype=np.float)



    old_analytical = np.zeros(6 * no_of_particles, dtype=np.float)

    """ Solver """

    for time_index, t0 in enumerate(time):
      print("Computing for TimeIndex = ", time_index)
      # print('\n \n')
      t0 = time[time_index]
      if (time_index == time.size - 1):
        break
      t1 = time[time_index + 1]
      t = [t0, t1]
      if (time_index == 0):
        initial_conditions = initial_conditions

      else:
        initial_conditions = old

      Jx, Jy, Jz = 0, 0, 0
      
      Ex_updated, Ey_updated, Ez_updated, Bx_updated, By_updated, Bz_updated = fdtd(Ex, Ey, Ez, Bx, By, Bz, speed_of_light, Lx, Ly, ghost_cells, Jx, Jy, Jz, dt)

      if(time_index==0):

        fracs_Ex_x, fracs_Ex_y = fraction_finder((x_initial), (y_initial), (x_right), (y_center))

        fracs_Ey_x, fracs_Ey_y = fraction_finder((x_initial), (y_initial), (x_center), (y_top))

        fracs_Ez_x, fracs_Ez_y = fraction_finder((x_initial), (y_initial), (x_center), (y_center))

        fracs_Bx_x, fracs_Bx_y = fraction_finder((x_initial), (y_initial), (x_center), (y_top))

        fracs_By_x, fracs_By_y = fraction_finder((x_initial), (y_initial), (x_right), (y_center))

        fracs_Bz_x, fracs_Bz_y = fraction_finder((x_initial), (y_initial), (x_right), (y_top))

      else:
        fracs_Ex_x, fracs_Ex_y = fraction_finder(x_coords, y_coords, x_right, y_center)

        fracs_Ey_x, fracs_Ey_y = fraction_finder(x_coords, y_coords, x_center, y_top)

        fracs_Ez_x, fracs_Ez_y = fraction_finder(x_coords, y_coords, x_center, y_center)

        fracs_Bx_x, fracs_Bx_y = fraction_finder(x_coords, y_coords, x_center, y_top)

        fracs_By_x, fracs_By_y = fraction_finder(x_coords, y_coords, x_right, y_center)

        fracs_Bz_x, fracs_Bz_y = fraction_finder(x_coords, y_coords, x_right, y_top)


      Ex_particle = af.signal.approx2(Ex, fracs_Ex_y, fracs_Ex_x)

      Ey_particle = af.signal.approx2(Ey, fracs_Ey_y, fracs_Ey_x)

      Ez_particle = af.signal.approx2(Ez, fracs_Ez_y, fracs_Ez_x)

      Bx_particle = af.signal.approx2(Bx, fracs_Bx_y, fracs_Bx_x)

      By_particle = af.signal.approx2(By, fracs_By_y, fracs_By_x)

      Bz_particle = af.signal.approx2(Bz, fracs_Bz_y, fracs_Bz_x)

      (x_coords, y_coords, z_coords, vel_x, vel_y, vel_z) = integrator(x_initial, y_initial, z_initial,\
                                                                       vel_x_initial, vel_y_initial, vel_z_initial, dt, \
                                                                       Ex_particle, Ey_particle, Ez_particle,\
                                                                       Bx_particle, By_particle, Bz_particle\
                                                                      )


      if (time_index == 0):
        initial_conditions_analytical = [ \
          af.sum(x_coords[0]),af.sum(y_coords[0]) , \
                                         af.sum((vel_x_initial[0]+vel_x[0])/2),af.sum ((vel_y_initial[0]+vel_y[0])/2)\
                                                ]
        # x,initial = x(n+1), vx_initial = avg(v(n+0.5,n+1.5)

        # all below from n = 1 (start is n = 0)
        # print('aasttarts',initial_conditions_analytical)
        initial_conditions_analytical = initial_conditions_analytical
        # print('22222', initial_conditions_analytical)
        position_analytical[time_index,0] = (initial_conditions_analytical[0]) # x
        position_analytical[time_index,1] = (initial_conditions_analytical[1]) # y
        velocity_analytical[time_index, 0] = (initial_conditions_analytical[2]) # vx
        velocity_analytical[time_index, 1] = (initial_conditions_analytical[3]) # vy

      else:
        initial_conditions_analytical = old_analytical

      # print('Hello', initial_conditions_analytical)
      sol_analytical = odeint(analytical, initial_conditions_analytical, t)

      (x_coords, vel_x, vel_y, vel_z) = wall_x(x_coords, vel_x, vel_y, vel_z)
      (y_coords, vel_x, vel_y, vel_z) = wall_y(y_coords, vel_x, vel_y, vel_z)
      (z_coords, vel_x, vel_y, vel_z) = wall_z(z_coords, vel_x, vel_y, vel_z)

      # print('vx, vy = ', vel_x, vel_y)

      if(sol_analytical[1, 0] > right_boundary):
        sol_analytical[1, 0]-=Lx
      if(sol_analytical[1, 0] < left_boundary):
        sol_analytical[1, 0]+=Lx


      if(sol_analytical[1, 1] > top_boundary):
        sol_analytical[1, 1]-=Ly
      if(sol_analytical[1, 1] < bottom_boundary):
        sol_analytical[1, 1]+=Ly

      # saving numerical data for position from n =1 timestep
      position_numerical[time_index, 0] = af.sum(x_coords[0])
      position_numerical[time_index, 1] = af.sum(y_coords[0])

      # saving numerical data for velocity from n =1 timestep

      if(time_index == 0) :
        velocity_numerical[time_index, 0] = af.sum(vel_x[ 0] + vel_x_initial[0])/2 # n+1.5 and n+0.5
        velocity_numerical[time_index, 1] = af.sum(vel_y[ 0] + vel_y_initial[0])/2
      else:
        velocity_numerical[time_index, 0] = af.sum(vel_x[ 0] + vel_x_initial[ 0])/2
        velocity_numerical[time_index, 1] = af.sum(vel_y[ 0] + vel_y_initial[ 0])/2


      Ex, Ey, Ez, Bx, By, Bz= Ex_updated, Ey_updated, Ez_updated, Bx_updated, By_updated, Bz_updated

      x_initial, y_initial, z_initial, vel_x_initial, vel_y_initial, vel_z_initial = x_coords, y_coords, z_coords, vel_x, vel_y, vel_z

      old_analytical = sol_analytical[1, :]
      sol_analytical = sol_analytical[1, :]

      # saving analytical data starting from n = 1 timestep for both x and y, THat is 1st entry is n =1 timestep data for all

      position_analytical[time_index + 1, 0] = sol_analytical[0]
      position_analytical[time_index + 1, 1] = sol_analytical[1]
      velocity_analytical[time_index + 1, 0] = sol_analytical[2]
      velocity_analytical[time_index + 1, 1] = sol_analytical[3]

    h5f = h5py.File('data_files/time/solution'+str(Nx)+'.h5', 'w')
    h5f.create_dataset('data_files/time/solution_dataset'+str(Nx), data=time)
    h5f.close()

    h5f = h5py.File('data_files/posa/solution'+str(Nx)+'.h5', 'w')
    h5f.create_dataset('data_files/posa/solution_dataset'+str(Nx), data=position_analytical)
    h5f.close()

    h5f = h5py.File('data_files/posn/solution'+str(Nx)+'.h5', 'w')
    h5f.create_dataset('data_files/posn/solution_dataset'+str(Nx), data=position_numerical)
    h5f.close()

    h5f = h5py.File('data_files/vela/solution'+str(Nx)+'.h5', 'w')
    h5f.create_dataset('data_files/vela/solution_dataset'+str(Nx), data=velocity_analytical)
    h5f.close()

    h5f = h5py.File('data_files/veln/solution'+str(Nx)+'.h5', 'w')
    h5f.create_dataset('data_files/veln/solution_dataset'+str(Nx), data=velocity_numerical)
    h5f.close()

  return 1



N = np.array( [32, 64, 128, 256 ] )
# N = np.array([10])


x = error(N, N)
print(x)
