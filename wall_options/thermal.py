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

#Here we complete import of all the variable from the parameters file

""" 
This file contains 3 functions, which define thermal B.C's in 3 directions
Depending upon the choice of the user, thermal boundary conditions may be set
to either of the x,y and z directions.

A thermal B.C means that a particle that encounters such a boundary will reflect
back with its component of vel perpendicular to the wall, away from it with 
its magnitude taking a value corresponding to the temperature of the wall
"""


def wall_x(x_coords, vel_x, vel_y, vel_z):

  collided_right = af.algorithm.where(x_coords>right_boundary)
  collided_left  = af.algorithm.where(x_coords<left_boundary)

  # Random variables used to assign the new velocities to the particles:
  
  R1 = af.randu(af.Array.elements(collided_right))
  R2 = af.randu(af.Array.elements(collided_right))
  R3 = af.randu(af.Array.elements(collided_right))
  
  # 1e-12 has been subtracted so as to avoid errors in zonal computation
  # Using our methods for zonal computation, a particle at the right boundary, would be assigned to the zone beyond the wall
  x_coords[collided_right] = right_boundary - 1e-12

  constant_multiply     = 2*T_right_wall*(boltzmann_constant/mass_particle)
  vel_x[collided_right] = af.arith.sqrt(-constant_multiply*af.arith.log(R1))*(-1)    
  vel_y[collided_right] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R3)
  vel_z[collided_right] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R3)

  R1 = af.randu(af.Array.elements(collided_left))
  R2 = af.randu(af.Array.elements(collided_left))
  R3 = af.randu(af.Array.elements(collided_left))

  x_coords[collided_left] = left_boundary

  constant_multiply    = 2*T_left_wall*(boltzmann_constant/mass_particle)
  vel_x[collided_left] = af.arith.sqrt(-constant_multiply*af.arith.log(R1))    
  vel_y[collided_left] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R3)
  vel_z[collided_left] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R3)

  return(x_coords, vel_x, vel_y, vel_z)

def wall_y(y_coords, vel_x, vel_y, vel_z):

  y_coords  = sol[no_of_particles:2*no_of_particles]

  collided_top = af.algorithm.where(y_coords>top_boundary)
  collided_bot = af.algorithm.where(y_coords<bottom_boundary)

  # Random variables used to assign the new velocities to the particles:
  R1 = af.randu(af.Array.elements(collided_top))
  R2 = af.randu(af.Array.elements(collided_top))
  R3 = af.randu(af.Array.elements(collided_top))
  
  # 1e-12 has been subtracted so as to avoid errors in zonal computation
  y_coords[collided_top] = top_boundary - 1e-12

  constant_multiply   = 2*T_top_wall*(boltzmann_constant/mass_particle)
  vel_x[collided_top] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R3)    
  vel_y[collided_top] = af.arith.sqrt(-constant_multiply*af.arith.log(R1))*(-1)
  vel_z[collided_top] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R3)

  R1 = af.randu(af.Array.elements(collided_bot))
  R2 = af.randu(af.Array.elements(collided_bot))
  R3 = af.randu(af.Array.elements(collided_bot))

  y_coords[collided_bot] = bottom_boundary

  constant_multiply   = 2*T_bot_wall*(boltzmann_constant/mass_particle)
  vel_x[collided_bot] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R3)    
  vel_y[collided_bot] = af.arith.sqrt(-constant_multiply*af.arith.log(R1))
  vel_z[collided_bot] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R3)

  return(y_coords, vel_x, vel_y, vel_z)

def wall_z(z_coords, vel_x, vel_y, vel_z):

  collided_front = af.algorithm.where(z_coords>front_boundary)
  collided_back  = af.algorithm.where(z_coords<back_boundary)

  # Random variables used to assign the new velocities to the particles:
  R1 = af.randu(af.Array.elements(collided_front))
  R2 = af.randu(af.Array.elements(collided_front))
  R3 = af.randu(af.Array.elements(collided_front))

  # 1e-12 has been subtracted so as to avoid errors in zonal computation
  z_coords[collided_front] = front_boundary - 1e-12

  constant_multiply     = 2*T_front_wall*(boltzmann_constant/mass_particle)
  vel_x[collided_front] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R3)    
  vel_y[collided_front] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R3)
  vel_z[collided_front] = af.arith.sqrt(-constant_multiply*af.arith.log(R1))*(-1)
  
  R1 = af.randu(af.Array.elements(collided_back))
  R2 = af.randu(af.Array.elements(collided_back))
  R3 = af.randu(af.Array.elements(collided_back))

  z_coords[collided_back] = back_boundary

  constant_multiply    = 2*T_back_wall*(boltzmann_constant/mass_particle)
  vel_x[collided_back] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R3)    
  vel_y[collided_back] = np.sqrt(constant_multiply)*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R3)
  vel_z[collided_back] = af.arith.sqrt(-constant_multiply*af.arith.log(R1))
  
  return(z_coords, vel_x, vel_y, vel_z)