import numpy as np
from scipy.special import erfinv
import h5py
import params
import arrayfire as af

"""Here we shall assign values as set in params"""

no_of_particles      = params.no_of_particles
choice_integrator    = params.choice_integrator
collision_operator   = params.collision_operator
arrayfire_backend    = params.arrayfire_backend

af.set_backend(arrayfire_backend)

if(collision_operator == "hardsphere"):
  scattering_distance = params.scattering_distance

elif(collision_operator == "potential-based"):
  potential_steepness     = params.potential_steepness
  potential_amplitude     = params.potential_amplitude
  order_finite_difference = params.order_finite_difference

elif(collision_operator == "montecarlo"):
  x_zones_montecarlo = params.x_zones_montecarlo
  y_zones_montecarlo = params.y_zones_montecarlo
  z_zones_montecarlo = params.x_zones_montecarlo

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
This model uses the HS model for scattering the particles.
By this model the particles are effectively scattered in the 
same way as to how one models collisions amongst billiard 
balls. By this model we shall condsider only 2-body collisions.
Multi-body collisions are treated as 2 body collisions, with the 
choice of the 2 bodies being random.
"""

def collision_operator(x_coords, y_coords, z_coords, vel_x, vel_y, vel_z):

  x_coords_1   = af.tile(x_coords, 1, no_of_particles)        
  x_coords_2   = af.tile(af.reorder(x_coords, 1), no_of_particles, 1)
  x_difference = x_coords_1 - x_coords_2

  y_coords_1   = af.tile(y_coords, 1, no_of_particles)        
  y_coords_2   = af.tile(af.reorder(y_coords, 1), no_of_particles, 1)
  y_difference = y_coords_1 - y_coords_2

  z_coords_1   = af.tile(z_coords, 1, no_of_particles)        
  z_coords_2   = af.tile(af.reorder(z_coords, 1), no_of_particles, 1)
  z_difference = z_coords_1 - z_coords_2

  distance = af.arith.sqrt(x_difference**2 + y_difference**2 + z_difference**2)
  rcap_x   = x_difference/distance
  rcap_y   = y_difference/distance
  rcap_z   = z_difference/distance

  for i in range(no_of_particles):
   
    indices = af.algorithm.where((distance[:,i]<0.01)-af.identity(no_of_particles, no_of_particles)[:,i])
    index   = indices[np.random.randint(0, af.Array.elements(indices))]
   
    p = (vel_x[i]*rcap_x[index, i]     + vel_y[i]*rcap_y[index, i]     + vel_z[i]*rcap_z[index, i] - \
         vel_x[index]*rcap_x[index, i] - vel_y[index]*rcap_y[index, i] - vel_z[index]*rcap_z[index, i])
    
    vel_x[i] = vel_x[i] - p*rcap_x
    vel_y[i] = vel_y[i] - p*rcap_y
    vel_z[i] = vel_z[i] - p*rcap_z
    
    vel_x[index] = vel_x[index] - p*rcap_x
    vel_y[index] = vel_y[index] - p*rcap_y
    vel_z[index] = vel_z[index] - p*rcap_z
   
  return(vel_x, vel_y, vel_z)