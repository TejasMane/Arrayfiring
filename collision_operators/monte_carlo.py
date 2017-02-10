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
This script incorporates the montecarlo method of scattering.
By this method we divide our region of interest into zones and 
scatter the particles using an MB distribution, corresponding to 
the temperature of that zone
"""

def collision_operator(x_coords, y_coords, z_coords, vel_x, vel_y, vel_z):
  
  x_zones_particle   = (x_zones_montecarlo/length_box_x) * (x_coords - left_boundary)
  y_zones_particle   = (y_zones_montecarlo/length_box_y) * (y_coords - bottom_boundary)
  z_zones_particle   = (z_zones_montecarlo/length_box_z) * (y_coords - back_boundary)
  
  x_zones_particle   = af.Array.as_type(x_zones_particle, af.Dtype.u16)
  y_zones_particle   = af.Array.as_type(y_zones_particle, af.Dtype.u16)
  z_zones_particle   = af.Array.as_type(z_zones_particle, af.Dtype.u16)

  zone               = z_zones_particle*(x_zones_montecarlo*y_zones_particle + x_zones_particle)
  zonecount          = af.randu(af.max(zone))
  
  temp = np.zeros(zonecount.size)

  for i in range(x_zones_montecarlo*y_zones_montecarlo*z_zones_montecarlo):
    indices      = af.algorithm.where(zone == i)
    temp[i]      = (1/3)*np.sum(vel_x[indices]**2 + vel_y[indices]**2 + vel_z[indices]**2)
    zonecount[i] = af.Array.elements(indices)
  
  temp=temp/zonecount
    
  for i in range(x_zones_montecarlo*y_zones_montecarlo*z_zones_montecarlo):
    
    indices = af.algorithm.where(zone == i)
    
    R1 = af.randu(zonecount[i])
    R2 = af.randu(zonecount[i])
    R3 = af.randu(zonecount[i])
    R4 = af.randu(zonecount[i])
    
    constant_multiply  = np.sqrt(2*boltzmann_constant*temp[i]/mass_particle)
    vel_x[indices]     = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R1)
    vel_y[indices]     = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R1)
    vel_z[indices]     = constant_multiply*af.arith.sqrt(-af.arith.log(R4))*af.arith.cos(2*np.pi*R3)

  return(vel_x, vel_y, vel_z)