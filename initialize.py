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
This file is used in providing the initial velocities and positions to the particles.
The distribution of your choice may be obtained by modifying the options that have been 
provided below. Although the choice of function must be changed from this file, the parameter 
change may also be made at params.py
"""

""" Initializing the positions for the particles """

initial_position_x = left_boundary   + length_box_x * af.randu(no_of_particles)
initial_position_y = bottom_boundary + length_box_y * af.randu(no_of_particles)
initial_position_z = back_boundary   + length_box_z * af.randu(no_of_particles)

""" Initializing velocities to the particles """

# Declaring the random variable which shall be used to sample velocities:
R1 = af.randu(no_of_particles)
R2 = af.randu(no_of_particles)
R3 = af.randu(no_of_particles)
R4 = af.randu(no_of_particles)

# Sampling velocities corresponding to Maxwell-Boltzmann distribution at T_initial
# For this we shall be using the Box-Muller transformation
constant_multiply  = np.sqrt(2*boltzmann_constant*T_initial/mass_particle)
initial_velocity_x = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R1)
initial_velocity_y = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R1)
initial_velocity_z = constant_multiply*af.arith.sqrt(-af.arith.log(R4))*af.arith.cos(2*np.pi*R3)

""" Time parameters for the simulation """

box_crossing_time_scale = (length_box_x/af.algorithm.max(initial_velocity_x))
final_time              = 20 * box_crossing_time_scale
dt                      = 0.001 * box_crossing_time_scale
time                    = np.arange(0, final_time, dt)

""" Writing the data to a file """

h5f = h5py.File('data_files/initial_conditions/initial_data.h5', 'w')
h5f.create_dataset('time',     data = time)
h5f.create_dataset('x_coords', data = initial_position_x)
h5f.create_dataset('y_coords', data = initial_position_y)
h5f.create_dataset('z_coords', data = initial_position_z)
h5f.create_dataset('vel_x',    data = initial_velocity_x)
h5f.create_dataset('vel_y',    data = initial_velocity_y)
h5f.create_dataset('vel_z',    data = initial_velocity_z)
h5f.close()