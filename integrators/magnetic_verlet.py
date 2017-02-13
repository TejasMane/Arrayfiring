import numpy as np
from scipy.special import erfinv
import h5py
import params

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

"""
This is the integrator that implements the Boris Algorithm.
Add details here
"""

def integrator(x_coords, y_coords, z_coords, vel_x, vel_y, vel_z, dt, Ex, Ey, Ez, Bx, By, Bz):

  x   = x_coords
  y   = y_coords
  z   = z_coords
  v_x = vel_x
  v_y = vel_y
  v_z = vel_z

  Ex = Ex
  Ey = Ey
  Ez = Ez
  Bx = Bx
  By = By
  Bz = Bz

  v_x_minus = v_x + (charge * Ex * dt) / (2 * mass_particle)
  v_y_minus = v_y + (charge * Ey * dt) / (2 * mass_particle)
  v_z_minus = v_z + (charge * Ez * dt) / (2 * mass_particle)

  t_magx    = (charge * Bx * dt) / (2 * mass_particle)
  t_magy    = (charge * By * dt) / (2 * mass_particle)
  t_magz    = (charge * Bz * dt) / (2 * mass_particle)

  vminus_cross_t_x = (v_y_minus * t_magz) - (v_z_minus * t_magy)
  vminus_cross_t_y = -((v_x_minus * t_magz) - (v_z_minus * t_magx))
  vminus_cross_t_z = (v_x_minus * t_magy) - (v_y_minus * t_magx)

  v_dashx = v_x_minus + vminus_cross_t_x
  v_dashy = v_y_minus + vminus_cross_t_y
  v_dashz = v_z_minus + vminus_cross_t_z

  t_mag = np.sqrt(t_magx ** 2 + t_magy ** 2 + t_magz ** 2)

  s_x = (2 * t_magx) / (1 + abs(t_mag ** 2))
  s_y = (2 * t_magy) / (1 + abs(t_mag ** 2))
  s_z = (2 * t_magz) / (1 + abs(t_mag ** 2))

  v_x_plus = v_x_minus + ((v_dashy * s_z) - (v_dashz * s_y))
  v_y_plus = v_y_minus + -((v_dashx * s_z) - (v_dashz * s_x))
  v_z_plus = v_z_minus + ((v_dashx * s_y) - (v_dashy * s_x))

  vel_x_new  = v_x_plus + (charge * Ex * dt) / (2 * mass_particle)
  vel_y_new  = v_y_plus + (charge * Ey * dt) / (2 * mass_particle)
  vel_z_new  = v_z_plus + (charge * Ez * dt) / (2 * mass_particle)

  # Using v at (n+0.5) dt to push x at (n)dt

  x_coords_new = x_coords + vel_x_new * dt
  y_coords_new = y_coords + vel_y_new * dt
  z_coords_new = z_coords + vel_z_new * dt

  return (x_coords_new, y_coords_new, z_coords_new,\
          vel_x_new   , vel_y_new   , vel_z_new \
         )
