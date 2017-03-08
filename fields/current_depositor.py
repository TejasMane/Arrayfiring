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

# charge b0 depositor
def charge_b0_depositor(x, y, x_grid, y_grid, ghost_cells, Lx, Ly):

  x_charge_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.u32)
  y_charge_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.u32)

  nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
  ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

  dx = Lx/nx
  dy = Ly/ny

  x_zone = (((af.abs(x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
  y_zone = (((af.abs(y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))

  indices = af.where(af.abs(x-x_grid[x_zone])<af.abs(x-x_grid[x_zone + 1]))

  if(indices.elements()>0):
    x_charge_zone[indices] = x_zone[indices]

  indices = af.where(af.abs(x-x_grid[x_zone])>=af.abs(x-x_grid[x_zone + 1]))

  if(indices.elements()>0):
    x_charge_zone[indices] = (x_zone[indices] + 1).as_type(af.Dtype.u32)

  indices = af.where(af.abs(y - y_grid[y_zone])<af.abs(y - y_grid[y_zone + 1]))

  if(indices.elements()>0):
    y_charge_zone[indices] = y_zone[indices]

  indices = af.where(af.abs(y - y_grid[y_zone])>=af.abs(y - y_grid[y_zone + 1]))

  if(indices.elements()>0):
    y_charge_zone[indices] = (y_zone[indices] +1).as_type(af.Dtype.u32)


  charge_by_dxdy = (charge/(dx*dy)).as_type(af.Dtype.f64)

  af.eval(y_current_zone, x_current_zone)
  af.eval(current_by_dxdy)

  return x_charge_zone, y_charge_zone, charge_by_dxdy

# b1 charge depositor
def charge_b1_depositor(charge, x, y, x_grid, y_grid, ghost_cells, Lx, Ly):

  number_of_particles = x.elements()

  x_current_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)
  y_current_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)

  nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
  ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

  dx = Lx/nx
  dy = Ly/ny

  x_zone = (((af.abs(x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
  y_zone = (((af.abs(y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))

  x_zone_plus = x_zone + 1
  y_zone_plus = y_zone + 1

  dy_by_delta_y = (1/dy) * (y-y_grid[y_zone])
  dy_by_delta_y_complement = 1 - dy_by_delta_y

  dx_by_delta_x = (1/dx) * (x-x_grid[x_zone])
  dx_by_delta_x_complement = 1 - dx_by_delta_x

  weight_corner1 = dy_by_delta_y_complement * dx_by_delta_x_complement
  weight_corner2 = dy_by_delta_y * dx_by_delta_x_complement
  weight_corner3 = dy_by_delta_y * dx_by_delta_x
  weight_corner4 = dy_by_delta_y_complement * dx_by_delta_x

  charge_by_dxdy = ((charge/(dx*dy))).as_type(af.Dtype.f64)

  corner1_charge   = weight_corner1 * charge_by_dxdy
  corner2_charge   = weight_corner2 * charge_by_dxdy
  corner3_charge   = weight_corner3 * charge_by_dxdy
  corner4_charge   = weight_corner4 * charge_by_dxdy

  all_corners_weighted_charge = af.join(0,corner1_charge, corner2_charge, corner3_charge, corner4_charge)

  x_current_zone[0 * number_of_particles : 1 * number_of_particles] = x_zone
  x_current_zone[1 * number_of_particles : 2 * number_of_particles] = x_zone
  x_current_zone[2 * number_of_particles : 3 * number_of_particles] = x_zone_plus
  x_current_zone[3 * number_of_particles : 4 * number_of_particles] = x_zone_plus

  y_current_zone[0 * number_of_particles : 1 * number_of_particles] = y_zone
  y_current_zone[1 * number_of_particles : 2 * number_of_particles] = y_zone_plus
  y_current_zone[2 * number_of_particles : 3 * number_of_particles] = y_zone_plus
  y_current_zone[3 * number_of_particles : 4 * number_of_particles] = y_zone

  af.eval(x_current_zone, y_current_zone)
  af.eval(all_corners_weighted_charge)

  return x_current_zone, y_current_zone, all_corners_weighted_charge



"""Current Deposition for B0 splines (Vectorized)"""

# current b0 depositor
def current_b0_depositor(charge, x, y, velocity_required, x_grid, y_grid, ghost_cells, Lx, Ly):

  x_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.u32)
  y_current_zone = af.data.constant(0,y.elements(), dtype=af.Dtype.u32)

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

  return x_current_zone, y_current_zone, current_by_dxdy



# b1 depositor Anti Clockwise d1 to d4 with d1 being the bottom left corner
# and d4 being the bottom right corner
def current_b1_depositor(charge, x, y, velocity_required, x_grid, y_grid, ghost_cells, Lx, Ly):

  number_of_particles = x.elements()

  x_current_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)
  y_current_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)

  nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
  ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

  dx = Lx/nx
  dy = Ly/ny

  x_zone = (((af.abs(x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
  y_zone = (((af.abs(y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))

  x_zone_plus = x_zone + 1
  y_zone_plus = y_zone + 1

  dy_by_delta_y = (1/dy) * (y-y_grid[y_zone])
  dy_by_delta_y_complement = 1 - dy_by_delta_y

  dx_by_delta_x = (1/dx) * (x-x_grid[x_zone])
  dx_by_delta_x_complement = 1 - dx_by_delta_x

  weight_corner1 = dy_by_delta_y_complement * dx_by_delta_x_complement
  weight_corner2 = dy_by_delta_y * dx_by_delta_x_complement
  weight_corner3 = dy_by_delta_y * dx_by_delta_x
  weight_corner4 = dy_by_delta_y_complement * dx_by_delta_x

  current_by_dxdy = ((charge/(dx*dy))*velocity_required).as_type(af.Dtype.f64)

  corner1_currents   = weight_corner1 * current_by_dxdy
  corner2_currents   = weight_corner2 * current_by_dxdy
  corner3_currents   = weight_corner3 * current_by_dxdy
  corner4_currents   = weight_corner4 * current_by_dxdy

  all_corners_weighted_current = af.join(0,corner1_currents, corner2_currents, corner3_currents, corner4_currents)

  x_current_zone[0 * number_of_particles : 1 * number_of_particles] = x_zone
  x_current_zone[1 * number_of_particles : 2 * number_of_particles] = x_zone
  x_current_zone[2 * number_of_particles : 3 * number_of_particles] = x_zone_plus
  x_current_zone[3 * number_of_particles : 4 * number_of_particles] = x_zone_plus

  y_current_zone[0 * number_of_particles : 1 * number_of_particles] = y_zone
  y_current_zone[1 * number_of_particles : 2 * number_of_particles] = y_zone_plus
  y_current_zone[2 * number_of_particles : 3 * number_of_particles] = y_zone_plus
  y_current_zone[3 * number_of_particles : 4 * number_of_particles] = y_zone

  af.eval(x_current_zone, y_current_zone)
  af.eval(all_corners_weighted_current)

  return x_current_zone, y_current_zone, all_corners_weighted_current


# # Test script for b1 depositor
# from fields.current_depositor import current_b1_depositor
# charge = 1
# x1 = af.Array([0.2,0.6])
# y1 = af.Array([0.2,0.6])
# velocity_required = af.Array([1.0,1.0])
# x_grid = af.Array([-1.0, 0.0, 1.0, 2.0])
# y_grid = af.Array([-1.0, 0.0, 1.0, 2.0])
# ghost_cells = 1
# Lx = 1.0
# Ly = 1.0
#
# print(current_b1_depositor(charge, x1, y1, velocity_required, x_grid, y_grid, ghost_cells, Lx, Ly))




def dcd(charge, no_of_particles, positions_x ,positions_y, positions_z, velocities_x, velocities_y, velocities_z, \
        x_center_grid, y_center_grid,shape_function, ghost_cells, Lx, Ly, dx, dy\
       ):

  # print('charge is ', charge)
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

  # print('Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices are ', Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices )
  # print('Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices are ', Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices)
  # Jx_test = af.data.constant(0, y_center_grid.elements(), x_center_grid.elements(), dtype=af.Dtype.f64)
  #
  #
  # for i in range(no_of_particles):
  #   Jx_test[af.sum(Jx_y_indices[i]), af.sum(Jx_x_indices[i])]  = Jx_test[af.sum(Jx_y_indices[i]), af.sum(Jx_x_indices[i])] +  Jx_values_at_these_indices[i]



  input_indices = (Jx_x_indices*(y_center_grid.elements()) + Jx_y_indices)

  Jx, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jx_values_at_these_indices)
  Jx = af.data.moddims(af.to_array(Jx), y_center_grid.elements(), x_center_grid.elements())




  Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices = shape_function( charge,positions_x, positions_y, velocities_y,\
                                                                          x_center_grid, y_top_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                         )


  input_indices = (Jy_x_indices*(y_center_grid.elements()) + Jy_y_indices)
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

  input_indices = (Jz_x_indices*(y_center_grid.elements()) + Jz_y_indices)
  Jz, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jz_values_at_these_indices)
  Jz = af.data.moddims(af.to_array(Jz),  y_center_grid.elements(), x_center_grid.elements())

  af.eval(Jx, Jy, Jz)

  return Jx, Jy, Jz



def Umeda_b1_deposition( charge, x, y, velocity_required_x, velocity_required_y,\
                         x_grid, y_grid, ghost_cells, Lx, Ly, dt\
                       ):

  # print('vx is ', velocity_required_x)
  # print('vy is ', velocity_required_y)



  x_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.u32)
  y_current_zone = af.data.constant(0,y.elements(), dtype=af.Dtype.u32)

  nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
  ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

  dx = Lx/nx
  dy = Ly/ny

  # print('dx is ', dx)
  # print('dy is ', dy)

  x_1 = (x - (velocity_required_x * dt)).as_type(af.Dtype.f64)
  x_2 = (x).as_type(af.Dtype.f64)

  y_1 = (y - (velocity_required_y * dt)).as_type(af.Dtype.f64)
  y_2 = (y).as_type(af.Dtype.f64)

  # print('y_1 is ', y_1)
  # print('y_2 is ', y_2)

  i_1 = ( ((af.abs( x_1 - af.sum(x_grid[0])))/dx) - ghost_cells).as_type(af.Dtype.u32)
  j_1 = ( ((af.abs( y_1 - af.sum(y_grid[0])))/dy) - ghost_cells).as_type(af.Dtype.u32)


  i_2 = ( ((af.abs( x_2 - af.sum(x_grid[0])))/dx) - ghost_cells).as_type(af.Dtype.u32)
  j_2 = ( ((af.abs( y_2 - af.sum(y_grid[0])))/dy) - ghost_cells).as_type(af.Dtype.u32)




  # print('j_1 is ', j_1)
  # print('j_2 is ', j_2)

  i_dx = dx * af.join(1, i_1, i_2)
  j_dy = dy * af.join(1, j_1, j_2)


  # print('j_dy is ', j_dy)


  i_dx_x_avg = af.join(1, af.max(i_dx,1), ((x_1+x_2)/2))
  j_dy_y_avg = af.join(1, af.max(j_dy,1), ((y_1+y_2)/2))

  # print('j_dy_y_avg is ', j_dy_y_avg)


  x_r_term_1 = dx + af.min(i_dx, 1)
  x_r_term_2 = af.max(i_dx_x_avg, 1)

  y_r_term_1 = dy + af.min(j_dy, 1)
  y_r_term_2 = af.max(j_dy_y_avg, 1)

  # print('y_r_term_1 is ', y_r_term_1)
  # print('y_r_term_2 is ', y_r_term_2)

  x_r_combined_term = af.join(1, x_r_term_1, x_r_term_2)
  y_r_combined_term = af.join(1, y_r_term_1, y_r_term_2)

  x_r = af.min(x_r_combined_term, 1)
  # print('y_r_combined_term is ', y_r_combined_term)
  y_r = af.min(y_r_combined_term, 1)

  # print('y_r is ', y_r)

  F_x_1 = charge * (x_r - x_1)/dt
  F_x_2 = charge * (x_2 - x_r)/dt

  # print('y_r, y_2 and y_1 is ', y_r, y_2, y_1)

  F_y_1 = charge * (y_r - y_1)/dt
  F_y_2 = charge * (y_2 - y_r)/dt

  # print('F_y_1', F_y_2)
  # print('F_y_2 is ',F_y_2)

  W_x_1 = (x_1 + x_r)/(2 * dx) - i_1
  W_x_2 = (x_2 + x_r)/(2 * dx) - i_2

  W_y_1 = (y_1 + y_r)/(2 * dy) - j_1
  W_y_2 = (y_2 + y_r)/(2 * dy) - j_2

  J_x_1_1 = (1/(dx * dy)) * (F_x_1 * (1 - W_y_1))
  J_x_1_2 = (1/(dx * dy)) * (F_x_1 * (W_y_1))

  J_x_2_1 = (1/(dx * dy)) * (F_x_2 * (1 - W_y_2))
  J_x_2_2 = (1/(dx * dy)) * (F_x_2 * (W_y_2))

  J_y_1_1 = (1/(dx * dy)) * (F_y_1 * (1 - W_x_1))
  J_y_1_2 = (1/(dx * dy)) * (F_y_1 * (W_x_1))

  J_y_2_1 = (1/(dx * dy)) * (F_y_2 * (1 - W_x_2))
  J_y_2_2 = (1/(dx * dy)) * (F_y_2 * (W_x_2))



  Jx_x_indices = af.join(0, i_1 + ghost_cells, i_1 + ghost_cells,\
                            i_2 + ghost_cells, i_2 + ghost_cells\
                        )
  Jx_y_indices = af.join(0, j_1 + ghost_cells, (j_1 + 1 + ghost_cells),\
                            j_2 + ghost_cells, (j_2 + 1 + ghost_cells)\
                        )
  Jx_values_at_these_indices = af.join(0, J_x_1_1, J_x_1_2, J_x_2_1, J_x_2_2)



  Jy_x_indices = af.join(0, i_1 + ghost_cells, (i_1 + 1 + ghost_cells),\
                            i_2 + ghost_cells, (i_2 + 1 + ghost_cells)\
                        )
  Jy_y_indices = af.join(0, j_1 + ghost_cells, j_1 + ghost_cells,\
                            j_2 + ghost_cells, j_2 + ghost_cells\
                        )
  Jy_values_at_these_indices = af.join(0, J_y_1_1, J_y_1_2, J_y_2_1, J_y_2_2)



  af.eval(Jx_x_indices, Jx_y_indices, Jy_x_indices, Jy_y_indices)
  af.eval(Jx_values_at_these_indices, Jy_values_at_these_indices)

  return Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,\
         Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices



def Umeda_2003(charge, no_of_particles, positions_x ,positions_y, positions_z, velocities_x, velocities_y, velocities_z, \
                x_center_grid, y_center_grid, ghost_cells, Lx, Ly, dx, dy, dt\
              ):

  x_right_grid = x_center_grid + dx/2
  y_top_grid = y_center_grid + dy/2

  elements = x_center_grid.elements()*y_center_grid.elements()

  Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,\
  Jy_x_indices, Jy_y_indices,\
   Jy_values_at_these_indices = Umeda_b1_deposition( charge,positions_x, positions_y, velocities_x,\
                                                     velocities_y, x_right_grid, y_center_grid,\
                                                     ghost_cells, Lx, Ly, dt\
                                                   )


  input_indices = (Jx_x_indices*(y_center_grid.elements()) + Jx_y_indices)

  Jx, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jx_values_at_these_indices)
  Jx = af.data.moddims(af.to_array(Jx), y_center_grid.elements(), x_center_grid.elements())

  input_indices = (Jy_x_indices*(y_center_grid.elements()) + Jy_y_indices)
  Jy, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jy_values_at_these_indices)
  Jy = af.data.moddims(af.to_array(Jy), y_center_grid.elements(), x_center_grid.elements())

  Jz_x_indices, Jz_y_indices, Jz_values_at_these_indices = current_b1_depositor( charge, positions_x, positions_y, velocities_z,\
                                                                          x_center_grid, y_center_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                         )
  # for i in range(no_of_particles):
  #   Jz[af.sum(Jz_x_indices[i]), af.sum(Jz_y_indices[i])] = Jz[af.sum(Jz_x_indices[i]), af.sum(Jz_y_indices[i])] + Jz_values_at_these_indices[i]

  input_indices = (Jz_x_indices*(y_center_grid.elements()) + Jz_y_indices)
  Jz, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jz_values_at_these_indices)
  Jz = af.data.moddims(af.to_array(Jz),  y_center_grid.elements(), x_center_grid.elements())

  af.eval(Jx, Jy, Jz)

  return Jx, Jy, Jz
