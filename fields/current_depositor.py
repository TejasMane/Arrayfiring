import params
import arrayfire as af
import time as timer
import numpy as np

# Here we complete import of all the variable from the parameters file
# Ensure Particles have not crossed the domain boundaries in before current deposition

# Returns current at n+1/2 when x_n and x_n+1 are provided

# Ex  = (x_right, y_center )
# Ey  = (x_center, y_top   )
# Ez  = (x_center, y_center)
# Bx  = (x_center, y_top   )
# By  = (x_right, y_center )
# Bz  = (x_right, y_top    )
# rho = (x_center, y_center ) # Not needed here
# Jx  = (x_right, y_center )
# Jy  = (x_center, y_top )
# Jz  = (x_center, y_center )

"""Charge Deposition for B0 splines (Have to vectorize)"""

# charge b0 depositor
def charge_b0_depositor(charge, x, y, velocity_required, x_grid, y_grid, ghost_cells, Lx, Ly):

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
def charge_b1_depositor(charge, x, y, velocity_required, x_grid, y_grid, ghost_cells, Lx, Ly):

  number_of_particles = x.elements()

  x_charge_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)
  y_charge_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)

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

  charge_by_dxdy = ((charge/(dx*dy)))

  corner1_charge   = weight_corner1 * charge_by_dxdy
  corner2_charge   = weight_corner2 * charge_by_dxdy
  corner3_charge   = weight_corner3 * charge_by_dxdy
  corner4_charge   = weight_corner4 * charge_by_dxdy

  all_corners_weighted_charge = af.join(0,corner1_charge, corner2_charge, corner3_charge, corner4_charge)

  x_charge_zone[0 * number_of_particles : 1 * number_of_particles] = x_zone
  x_charge_zone[1 * number_of_particles : 2 * number_of_particles] = x_zone
  x_charge_zone[2 * number_of_particles : 3 * number_of_particles] = x_zone_plus
  x_charge_zone[3 * number_of_particles : 4 * number_of_particles] = x_zone_plus

  y_charge_zone[0 * number_of_particles : 1 * number_of_particles] = y_zone
  y_charge_zone[1 * number_of_particles : 2 * number_of_particles] = y_zone_plus
  y_charge_zone[2 * number_of_particles : 3 * number_of_particles] = y_zone_plus
  y_charge_zone[3 * number_of_particles : 4 * number_of_particles] = y_zone

  af.eval(x_charge_zone, y_charge_zone)
  af.eval(all_corners_weighted_charge)

  return x_charge_zone, y_charge_zone, all_corners_weighted_charge

def direct_charge_deposition(charge, no_of_particles, positions_x ,positions_y,\
                             positions_z, velocities_x, velocities_y, velocities_z, \
                             x_center_grid, y_center_grid,shape_function, \
                             ghost_cells, Lx, Ly, dx, dy\
                            ):

  elements = x_center_grid.elements()*y_center_grid.elements()

  rho_x_indices, \
  rho_y_indices, \
  rho_values_at_these_indices = shape_function(charge,positions_x, positions_y,\
                                               velocities_x, x_center_grid, y_center_grid,\
                                               ghost_cells, Lx, Ly\
                                              )

  input_indices = (rho_x_indices*(y_center_grid.elements()) + rho_y_indices)

  rho, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=rho_values_at_these_indices)
  rho = af.data.moddims(af.to_array(rho), y_center_grid.elements(), x_center_grid.elements())

  af.eval(rho)

  return rho

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


def dcd(charge, no_of_particles, positions_x ,positions_y, positions_z, velocities_x, velocities_y, velocities_z, \
        x_center_grid, y_center_grid,shape_function, ghost_cells, Lx, Ly, dx, dy\
       ):

  # print('charge is ', charge)
  x_right_grid = x_center_grid + dx/2
  y_top_grid = y_center_grid + dy/2

  elements = x_center_grid.elements()*y_center_grid.elements()

  Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices = shape_function( charge,positions_x, positions_y, velocities_x,\
                                                                          x_right_grid, y_center_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                         )


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

  Jz_x_indices, Jz_y_indices, Jz_values_at_these_indices = shape_function( charge, positions_x, positions_y, velocities_z,\
                                                                          x_center_grid, y_center_grid,\
                                                                          ghost_cells, Lx, Ly\
                                                                         )

  input_indices = (Jz_x_indices*(y_center_grid.elements()) + Jz_y_indices)
  Jz, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jz_values_at_these_indices)
  Jz = af.data.moddims(af.to_array(Jz),  y_center_grid.elements(), x_center_grid.elements())

  af.eval(Jx, Jy, Jz)

  return Jx, Jy, Jz



def Umeda_b1_deposition( charge, x, y, velocity_required_x, velocity_required_y,\
                         x_grid, y_grid, ghost_cells, Lx, Ly, dt\
                       ):


  x_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.u32)
  y_current_zone = af.data.constant(0,y.elements(), dtype=af.Dtype.u32)

  nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
  ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

  dx = Lx/nx
  dy = Ly/ny

  x_1 = (x - (velocity_required_x * dt)).as_type(af.Dtype.f64)
  x_2 = (x).as_type(af.Dtype.f64)

  y_1 = (y - (velocity_required_y * dt)).as_type(af.Dtype.f64)
  y_2 = (y).as_type(af.Dtype.f64)


  i_1 = ( ((af.abs( x_1 - af.sum(x_grid[0])))/dx) - ghost_cells).as_type(af.Dtype.u32)
  j_1 = ( ((af.abs( y_1 - af.sum(y_grid[0])))/dy) - ghost_cells).as_type(af.Dtype.u32)


  i_2 = ( ((af.abs( x_2 - af.sum(x_grid[0])))/dx) - ghost_cells).as_type(af.Dtype.u32)
  j_2 = ( ((af.abs( y_2 - af.sum(y_grid[0])))/dy) - ghost_cells).as_type(af.Dtype.u32)

  i_dx = dx * af.join(1, i_1, i_2)
  j_dy = dy * af.join(1, j_1, j_2)

  i_dx_x_avg = af.join(1, af.max(i_dx,1), ((x_1+x_2)/2))
  j_dy_y_avg = af.join(1, af.max(j_dy,1), ((y_1+y_2)/2))

  x_r_term_1 = dx + af.min(i_dx, 1)
  x_r_term_2 = af.max(i_dx_x_avg, 1)

  y_r_term_1 = dy + af.min(j_dy, 1)
  y_r_term_2 = af.max(j_dy_y_avg, 1)

  x_r_combined_term = af.join(1, x_r_term_1, x_r_term_2)
  y_r_combined_term = af.join(1, y_r_term_1, y_r_term_2)

  x_r = af.min(x_r_combined_term, 1)
  y_r = af.min(y_r_combined_term, 1)

  F_x_1 = charge * (x_r - x_1)/dt
  F_x_2 = charge * (x_2 - x_r)/dt

  F_y_1 = charge * (y_r - y_1)/dt
  F_y_2 = charge * (y_2 - y_r)/dt

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

  input_indices = (Jz_x_indices*(y_center_grid.elements()) + Jz_y_indices)
  Jz, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jz_values_at_these_indices)
  Jz = af.data.moddims(af.to_array(Jz),  y_center_grid.elements(), x_center_grid.elements())

  af.eval(Jx, Jy, Jz)

  return Jx, Jy, Jz
