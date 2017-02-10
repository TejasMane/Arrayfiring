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
This script is used to test out how particles may be 
scattered by utilizing the fact that collisions are modelled 
by the means of a potential that acts in short range distances between
particles.
"""

def potential(distance):
  potential= (potential_amplitude/2) * ( -1 * np.tanh(potential_steepness*distance) + 1)
  return(potential)

# This function returns the value of potential gradient
# Here 'x' denotes the location where gradient is to be computed
# Here 'a' is the parameter which controls the steepness of the potential function

def potential_gradient(x, order):

  # We shall use the finite difference method to obtain the gradient of potential
  # There are several schemes of finite differencing with different orders of accuracy
  # Described below are the schemes used
  # In the below description O(deltaX) is the big-O notation 
  # In all the schemes, we have taken deltaX = 1e-10

  # In an order one scheme. The gradient of a quantity f w.r.t may be written as:
  # df/dx = ((f_{i+1} - f{i})/deltaX) + O(deltaX) 
  order1 = 1e10*(  potential(x+1e-10) - potential(x)     )

  # In an order one scheme. The gradient of a quantity f w.r.t may be written as:
  # df/dx = ((f_{i+1} - f{i-1})/2*deltaX) + O(deltaX^2)
  order2 = 5e9 *(  potential(x+1e-10)  - potential(x-1e-10))
  
  # In an order one scheme. The gradient of a quantity f w.r.t may be written as:
  # df/dx = (8*(f_{i+1} - f{i-1} + f_{i-2} - f_{i+2})/12*deltaX) + O(deltaX^4)
  order4 = (8  *(  potential(x+1e-10)  - potential(x-1e-10)) \
                 + potential(x-2e-10)  - potential(x+2e-10)  \
           ) /(12e-10)

  if   (order==4):
    return(order4)

  elif (order==2):
    return(order2)

  elif (order==1):
    return(order1)

  else:
    print("Error, order passed as argument not defined!")
    exit()

# This function calculates and returns the potential energy of the entire system
# This is done by summing over all the potentials of the particles in the system

def calculate_potential_energy(x_coords, y_coords, z_coords):

  # The following steps will generate a matrix, in which i-th row contains 
  # The difference in the x-coordinates of all particles - x-coordinate of i-th particle
    
    x_coords_1   = af.tile(x_coords, 1, no_of_particles)        
    x_coords_2   = af.tile(af.reorder(x_coords, 1), no_of_particles, 1)
    x_difference = x_coords_1 - x_coords_2
   
    y_coords_1   = af.tile(y_coords, 1, no_of_particles)        
    y_coords_2   = af.tile(af.reorder(y_coords, 1), no_of_particles, 1)
    y_difference = y_coords_1 - y_coords_2
   
    z_coords_1   = af.tile(z_coords, 1, no_of_particles)        
    z_coords_2   = af.tile(af.reorder(z_coords, 1), no_of_particles, 1)
    z_difference = z_coords_1 - z_coords_2


  # dist is [N x N]. dis[i, j] is the distance between particle i and particle j.
  dist = af.arith.sqrt(x_difference**2 + y_difference**2 + z_difference**2)

  # potential(a,dist) will return a [N X N] matrix, where
  # potential[i,j] is the pair potential between particles i and j
  # Summing over all potentials in the system will give us the total potential energy
  # Of the system at a particular time-step
  potential_energy = 0.5*(af.sum(potential(dist)-(potential_amplitude/2)*af.identity(no_of_particles, no_of_particles)))
  return(potential_energy)