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
This file contains 3 functions, which define hardwall B.C's in 3 directions
Depending upon the choice of the user, hardwall boundary conditions may be set
to either of the x,y and z directions.

A hardwall B.C means that a particle that encounters such a boundary will reflect
back with its component of velocity perpendicular to the wall being the same in
magnitude, but opposite in direction
"""

def wall_x(x_coords, vel_x, vel_y, vel_z):

  collided_right = af.algorithm.where(x_coords > right_boundary)
  collided_left  = af.algorithm.where(x_coords < left_boundary)

  x_coords[collided_left] = left_boundary
  vel_x[collided_left]    = vel_x[collided_left]*(-1)    

  x_coords[collided_right] = right_boundary
  vel_x[collided_right]    = vel_x[collided_right]*(-1)    

  return(x_coords, vel_x, vel_y, vel_z)

def wall_y(y_coords, vel_x, vel_y, vel_z):

  collided_top = af.algorithm.where(y_coords > top_boundary)
  collided_bot = af.algorithm.where(y_coords < bottom_boundary)

  y_coords[collided_bot] = botom_boundary
  vel_y[collided_bot]    = vel_y[collided_bot]*(-1)    

  y_coords[collided_top] = top_boundary
  vel_y[collided_top]    = vel_y[collided_top]*(-1)    

  return(y_coords, vel_x, vel_y, vel_z)

def wall_z(z_coords, vel_x, vel_y, vel_z):

  collided_front = af.algorithm.where(z_coords > front_boundary)
  collided_back  = af.algorithm.where(z_coords < back_boundary)

  z_coords[collided_back] = back_boundary
  vel_z[collided_back]    = vel_z[collided_back]*(-1)    

  z_coords[collided_front] = front_boundary
  vel_z[collided_front]    = vel_z[collided_front]*(-1)    

  return(z_coords, vel_x, vel_y, vel_z)