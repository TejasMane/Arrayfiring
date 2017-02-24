import params
import arrayfire as af
import time as timer
import numpy as np

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


def charge_b0_depositor(x, y, x_grid, y_grid, ghost_cells, Lx, Ly):

  x_charge_zone = af.data.constant(0, x.elements(), dtype=af.Dtype.u32)
  y_charge_zone = af.data.constant(0, x.elements(), dtype=af.Dtype.u32)

  nx = ((x_grid.elements()) - 1 - 2 * ghost_cells)  # number of zones
  ny = ((y_grid.elements()) - 1 - 2 * ghost_cells)  # number of zones

  dx = Lx / nx
  dy = Ly / ny

  x_zone = (((af.abs(x - af.sum(x_grid[0]))) / dx).as_type(af.Dtype.u32))
  y_zone = (((af.abs(y - af.sum(y_grid[0]))) / dy).as_type(af.Dtype.u32))

  indices = af.where(af.abs(x - x_grid[x_zone]) < af.abs(x - x_grid[x_zone + 1]))

  if (indices.elements() > 0):
    x_charge_zone[indices] = x_zone[indices]

  indices = af.where(af.abs(x - x_grid[x_zone]) >= af.abs(x - x_grid[x_zone + 1]))

  if (indices.elements() > 0):
    x_charge_zone[indices] = (x_zone[indices] + 1).as_type(af.Dtype.u32)

  indices = af.where(af.abs(y - y_grid[y_zone]) < af.abs(y - y_grid[y_zone + 1]))

  if (indices.elements() > 0):
    y_charge_zone[indices] = y_zone[indices]

  indices = af.where(af.abs(y - y_grid[y_zone]) >= af.abs(y - y_grid[y_zone + 1]))

  if (indices.elements() > 0):
    y_charge_zone[indices] = (y_zone[indices] + 1).as_type(af.Dtype.u32)

  charge_by_dxdy = ((charge / (dx * dy))).as_type(af.Dtype.f64)

  af.eval(y_charge_zone, x_charge_zone)
  af.eval(charge_by_dxdy)

  return y_charge_zone, x_charge_zone, charge_by_dxdy

  return y_charge_zone, x_charge_zone, charge_by_dxdy


"""Current Deposition for B0 splines (Vectorized)"""


def current_b0_depositor(charge, x, y, velocity_required, x_grid, y_grid, ghost_cells, Lx, Ly):
  
  x_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.u32)
  y_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.u32)

  nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
  ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

  dx = Lx/nx
  dy = Ly/ny

  x_zone = (((af.abs(x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
  y_zone = (((af.abs(y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))

  indices = af.where(af.abs(x-x_grid[x_zone])<af.abs(x-x_grid[x_zone + 1]))

  if(indices.elements()>0):
    x_current_zone[indices] = x_zone[indices]

  indices = af.where(af.abs(x-x_grid[x_zone])>=af.abs(x-x_grid[x_zone + 1]))

  if(indices.elements()>0):
    x_current_zone[indices] = (x_zone[indices] + 1).as_type(af.Dtype.u32)

  indices = af.where(af.abs(y - y_grid[y_zone])<af.abs(y - y_grid[y_zone + 1]))

  if(indices.elements()>0):
    y_current_zone[indices] = y_zone[indices]

  indices = af.where(af.abs(y - y_grid[y_zone])>=af.abs(y - y_grid[y_zone + 1]))

  if(indices.elements()>0):
    y_current_zone[indices] = (y_zone[indices] +1).as_type(af.Dtype.u32)

    
  current_by_dxdy = ((charge/(dx*dy))*velocity_required).as_type(af.Dtype.f64)

  af.eval(y_current_zone, x_current_zone)
  af.eval(current_by_dxdy)

  
  return y_current_zone, x_current_zone, current_by_dxdy




def dcd(charge, no_of_particles, positions_x ,positions_y, positions_z, velocities_x, velocities_y, velocities_z, \
        x_center_grid, y_center_grid,shape_function, ghost_cells, Lx, Ly, dx, dy\
       ):


  x_right_grid = x_center_grid + dx/2
  y_top_grid = y_center_grid + dy/2

  elements = x_center_grid.elements()*y_center_grid.elements()
  # Jx = af.data.constant(0, x_center_grid.elements(), y_center_grid.elements(), dtype=af.Dtype.f64)
  # Jy = af.data.constant(0, x_center_grid.elements(), y_center_grid.elements(), dtype=af.Dtype.f64)
  # Jz = af.data.constant(0, x_center_grid.elements(), y_center_grid.elements(), dtype=af.Dtype.f64)

  Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices = shape_function( charge,positions_x, positions_y, velocities_x,\
                                                                          x_right_grid, y_center_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                         )

  # Jx_test = af.data.constant(0, x_center_grid.elements(), y_center_grid.elements(), dtype=af.Dtype.f64)


  # for i in range(no_of_particles):
  #   Jx[af.sum(Jx_x_indices[i]), af.sum(Jx_y_indices[i])] = Jx[af.sum(Jx_x_indices[i]), af.sum(Jx_y_indices[i])] +  Jx_values_at_these_indices[i]



  input_indices = (Jx_y_indices*(x_center_grid.elements()) + Jx_x_indices)
  Jx, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jx_values_at_these_indices)
  Jx = af.data.moddims(af.to_array(Jx), y_center_grid.elements(), x_center_grid.elements())


  Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices = shape_function( charge,positions_x, positions_y, velocities_y,\
                                                                          x_center_grid, y_top_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                         )


  input_indices = (Jy_y_indices*(x_center_grid.elements()) + Jy_x_indices)
  Jy, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jy_values_at_these_indices)
  Jy = af.data.moddims(af.to_array(Jy), y_center_grid.elements(), x_center_grid.elements())


  # for i in range(no_of_particles):
  #   Jy[af.sum(Jy_x_indices[i]), af.sum(Jy_y_indices[i])] = Jy[af.sum(Jy_x_indices[i]), af.sum(Jy_y_indices[i])]+ Jy_values_at_these_indices[i]

  Jz_x_indices, Jz_y_indices, Jz_values_at_these_indices = shape_function( charge, positions_x, positions_y, velocities_z,\
                                                                          x_center_grid, y_center_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                         )
  # for i in range(no_of_particles):
  #   Jz[af.sum(Jz_x_indices[i]), af.sum(Jz_y_indices[i])] = Jz[af.sum(Jz_x_indices[i]), af.sum(Jz_y_indices[i])] + Jz_values_at_these_indices[i]

  input_indices = (Jz_y_indices*(x_center_grid.elements()) + Jz_x_indices)
  Jz, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jz_values_at_these_indices)
  Jz = af.data.moddims(af.to_array(Jz),  y_center_grid.elements(), x_center_grid.elements())

  af.eval(Jx, Jy, Jz)

  return Jx, Jy, Jz



