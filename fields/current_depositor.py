import params
import numpy as np
from wall_options.EM_periodic import periodic
from fields.fdtd import fdtd
import arrayfire as af


"""Here we shall re-assign values as set in params"""

no_of_particles      = params.no_of_particles
choice_integrator    = params.choice_integrator
collision_operator   = params.collision_operator

plot_spatial_temperature_profile = params.plot_spatial_temperature_profile

if(plot_spatial_temperature_profile == "true"):
  x_zones_temperature = params.x_zones_temperature
  y_zones_temperature = params.y_zones_temperature

elif(collision_operator == "potential-based"):
  potential_steepness     = params.potential_steepness
  potential_amplitude     = params.potential_amplitude
  order_finite_difference = params.order_finite_difference

elif(collision_operator == "montecarlo"):
  x_zones_montecarlo = params.x_zones_montecarlo
  y_zones_montecarlo = params.y_zones_montecarlo
  
mass_particle      = params.mass_particle
boltzmann_constant = params.boltzmann_constant
T_initial          = params.T_initial
wall_condition_x   = params.wall_condition_x
wall_condition_y   = params.wall_condition_y
wall_condition_z   = params.wall_condition_z

if(wall_condition_x == "thermal"):
  T_left_wall  = params.T_left_wall
  T_right_wall = params.T_right_wall

if(wall_condition_y == "thermal"):
  T_top_wall = params.T_top_wall
  T_bot_wall = params.T_bot_wall

if(wall_condition_z == "thermal"):
  T_front_wall = params.T_front_wall
  T_back_wall  = params.T_back_wall

fields_enabled   = params.fields_enabled

if(fields_enabled == "true"):
  spread            = params.spread
  ghost_cells       = params.ghost_cells
  speed_of_light    = params.speed_of_light
  charge            = params.charge
  x_zones_field     = params.x_zones_field
  y_zones_field     = params.y_zones_field

left_boundary    = params.left_boundary
right_boundary   = params.right_boundary
length_box_x     = params.length_box_x

bottom_boundary  = params.bottom_boundary
top_boundary     = params.top_boundary
length_box_y     = params.length_box_y

back_boundary    = params.back_boundary
front_boundary   = params.front_boundary
length_box_z     = params.length_box_z

# Here we complete import of all the variable from the parameters file
# Ensure Particles have not crossed the domain boundaries in before current deposition

# Returns current at n+1/2 when x_n and x_n+1 are provided

# Ex = (x_right, y_center )
# Ey = (x_center, y_top   )
# Ez = (x_center, y_center)
# Bx = (x_center, y_top   )
# By = (x_right, y_center )
# Bz = (x_right, y_top    )
# rho = (x_center, y_top ) # Not needed here
# Jx = (x_right, y_center )
# Jy = (x_center, y_top )
# Jz = (x_center, y_center )


"""Charge Deposition for B0 splines (Have to vectorize)"""

def charge_b0_depositor(x, y, x_grid, y_grid, J, ghost_cells, Lx, Ly):
  x_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.f64)
  y_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.f64)
  charge_by_dxdy = af.data.constant(0,x.elements(), dtype=af.Dtype.f64)
  
  for i in range(x.elements()):
    n = ((x_grid.elements) - 1 - 2 * ghost_cells)  # number of zones

    dx = Lx/nx
    dy = Ly/ny

    x_zone = int(af.sum(nx * (x[i] - x_grid[0]))/Lx)  # indexing from zero itself
    y_zone = int(af.sum(ny * (y[i] - y_grid[0]))/Ly)

    if(af.arith.abs(x[i]-x_grid[x_zone])<af.arith.abs(x[i]-x_grid[x_zone + 1])):
      x_current_zone[i] = x_zone
    else:
      x_current_zone[i] = x_zone +1


    if(af.arith.abs(y[i] - y_grid[y_zone])<af.arith.abs(y[i] - y_grid[y_zone + 1])):
      y_current_zone[i] = y_zone
    else:
      y_current_zone[i] = y_zone +1
    
    charge_by_dxdy[i] = charge[i]/(dx*dy)
    
  return y_current_zone,x_current_zone,((charge/(dx*dy)))


#charge_b0_depositor = np.vectorize(charge_b0_depositor, excluded=(['x_grid', 'y_grid', 'J','ghost_cells', 'Lx', 'Ly']))

# Example of usage Charge depositor

# x = np.array([0.9, 0.1])
# y = np.array([0.2, 0.6])
#
# x_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
# y_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#
# rho = np.matrix('0 0 0 0 0;0 0 0 0 0;0 0 0 0 0;0 0 0 0 0')
#
# m,n,o = charge_b0_depositor(x = [x], y= [y], x_grid = x_grid, y_grid = y_grid, J = rho, ghost_cells = ghost_cells, Lx = Lx, Ly = Ly   )
#

# print(m)
# print(n)
# print(o)



"""Current Deposition for B0 splines (Have to vectorize)"""

def current_b0_depositor(charge, x, y, velocity_required, x_grid, y_grid, ghost_cells, Lx, Ly):

  x_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.f64)
  y_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.f64)
  current_by_dxdy = af.data.constant(0,x.elements(), dtype=af.Dtype.f64)

  nx = ((x_grid.elements()) - 1 - 2 * ghost_cells)  # number of zones
  ny = ((y_grid.elements()) - 1 - 2 * ghost_cells)  # number of zones

  dx = Lx/nx
  dy = Ly/ny



  for i in range(x.elements()):

    x_zone = int(af.sum(nx * (x[i] - x_grid[0]))/Lx)  # indexing from zero itself
    y_zone = int(af.sum(ny * (y[i] - y_grid[0]))/Ly)

    if(af.abs(x[i]-x_grid[x_zone])<af.abs(x[i]-x_grid[x_zone + 1])):
      x_current_zone[i] = x_zone
    else:
      x_current_zone[i] = x_zone +1

    if(af.abs(y[i] - y_grid[y_zone])<af.abs(y[i] - y_grid[y_zone + 1])):
      y_current_zone[i] = y_zone
    else:
      y_current_zone[i] = y_zone +1
    current_by_dxdy = ((charge/(dx*dy))*velocity_required[i])
    
  return y_current_zone,x_current_zone,((charge/(dx*dy))*velocity_required)


#current_b0_depositor = np.vectorize(current_b0_depositor, excluded=(['charge','x_grid', 'y_grid', 'ghost_cells', 'Lx', 'Ly']))

# Example of usage Current depositor

# x = np.array([0.2, 0.6])
# y = np.array([0.2, 0.6])
# v = np.array([5, 5])
#
# x_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#
# y_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#
# J = np.matrix('0 0 0 0 0;0 0 0 0 0;0 0 0 0 0;0 0 0 0 0')
#
# i,j,k = np.array(current_b0_depositor(charge = 1, x = [x], y= [y], velocity_required = [v], x_grid = x_grid, y_grid = y_grid, ghost_cells = 1, Lx = 1, Ly = 1   ))
#
# # print(i,'\n \n')
#
# print('The i shape ', i.shape)
# print('The i  first element ', i[0,0])

# def fun(f):
#
#
#     x = np.array([0.2, 0.6])
#     y = np.array([0.2, 0.6])
#     v = np.array([5, 5])
#
#     x_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#
#     y_grid = np.array([-0.5, 0, 0.5, 1, 1.5])
#
#     e,b,c = f(charge = 1, x = [x], y= [y], velocity_required = [v], x_grid = x_grid, y_grid = y_grid, ghost_cells = 1, Lx = 1, Ly = 1   )
#     return e,b,c
#
# print(fun(current_b0_depositor))

def dcd(charge, no_of_particles, positions_plus_half ,velocities_plus_half, x_center_grid, y_center_grid,shape_function, ghost_cells, Lx, Ly, dx, dy):

  x_right_grid = x_center_grid + dx
  y_top_grid = y_center_grid + dy

  positions_x = positions_plus_half[:no_of_particles]
  positions_y = positions_plus_half[no_of_particles:2*no_of_particles]

  velocities_x = velocities_plus_half[:no_of_particles]
  velocities_y = velocities_plus_half[no_of_particles:2*no_of_particles]
  velocities_z = velocities_plus_half[2*no_of_particles:3*no_of_particles]

  Jx = af.data.constant(0, x_center_grid.elements(), y_center_grid.elements(), dtype=af.Dtype.f64)
  Jy = af.data.constant(0, x_center_grid.elements(), y_center_grid.elements(), dtype=af.Dtype.f64)
  Jz = af.data.constant(0, x_center_grid.elements(), y_center_grid.elements(), dtype=af.Dtype.f64)

  Jx_x_indice, Jx_y_indices, Jx_values_at_these_indices = shape_function( charge,positions_x, positions_y, velocities_x,\
                                                                          x_right_grid, y_center_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                        )

  for i in range(no_of_particles):
    Jx[af.sum(Jx_x_indice[0,i]), af.sum(Jx_y_indices[0,i])] = Jx_values_at_these_indices[0,i]

  Jy_x_indice, Jy_y_indices, Jy_values_at_these_indices = shape_function( charge,positions_x, positions_y, velocities_y,\
                                                                          x_center_grid, y_top_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                        )

  for i in range(no_of_particles):
    Jy[af.sum(Jy_x_indice[0,i]), af.sum(Jy_y_indices[0,i])] = Jy_values_at_these_indices[0,i]

  Jz_x_indice, Jz_y_indices, Jz_values_at_these_indices = shape_function( charge, positions_x, positions_y, velocities_z,\
                                                                          x_center_grid, y_center_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                        )

  for i in range(no_of_particles):
    Jy[af.sum(Jz_x_indice[0,i]), af.sum(Jz_y_indices[0,i])] = Jz_values_at_these_indices[0,i]

  return Jx, Jy, Jz
