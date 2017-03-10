import numpy as np
import h5py
import params
import arrayfire as af
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
  spread             = params.spread
  ghost_cells        = params.ghost_cells
  speed_of_light     = params.speed_of_light
  charge             = params.charge
  x_zones_field      = params.x_zones_field
  y_zones_field      = params.y_zones_field
  k_fourier          = params.k_fourier
  Amplitude_perturbed= params.Amplitude_perturbed
Amplitude_perturbed= params.Amplitude_perturbed
left_boundary    = params.left_boundary
right_boundary   = params.right_boundary
length_box_x     = params.length_box_x

bottom_boundary  = params.bottom_boundary
top_boundary     = params.top_boundary
length_box_y     = params.length_box_y

back_boundary    = params.back_boundary
front_boundary   = params.front_boundary
length_box_z     = params.length_box_z

"""
This file is used in providing the initial velocities and positions to the particles.
The distribution of your choice may be obtained by modifying the options that have been
provided below. Although the choice of function must be changed from this file, the parameter
change may also be made at params.py
"""

""" Initializing the positions for the particles """

# initial_position_x = left_boundary   + length_box_x * af.randu(no_of_particles)
# initial_position_y = bottom_boundary + length_box_y * af.randu(no_of_particles)
# initial_position_z = back_boundary   + length_box_z * af.randu(no_of_particles)
"""Initializing x positions here"""

# Might not initialize correctly for some divisions


# x_divisions_perturbed = 100
# length_of_box_x         = right_boundary - left_boundary
# initial_position_x=np.zeros(no_of_particles)
# last=0
# next=0
# for i in range(x_divisions_perturbed):
#    next=last+(no_of_particles*Amplitude_perturbed*np.sin(2*i*np.pi/x_divisions_perturbed)/x_divisions_perturbed)+(no_of_particles/x_divisions_perturbed)
#    initial_position_x[int(round(last)):(int(round(next))-1)] = length_of_box_x*(i+1)/(x_divisions_perturbed+1)
#    last=next
initial_position_x = 0.25 + left_boundary   + 0.5 * length_box_x * af.randu(no_of_particles)
initial_position_y = 0.25 + bottom_boundary + 0.5 * length_box_y * af.randu(no_of_particles)
initial_position_z = 0.25 + back_boundary   + 0.5 * length_box_z * af.randu(no_of_particles)


""" Initializing velocities to the particles """

# Declaring the random variable which shall be used to sample velocities:

# R1 = af.randu(no_of_particles)
# R2 = af.randu(no_of_particles)
# R3 = af.randu(no_of_particles)
# R4 = af.randu(no_of_particles)

R1 = np.random.rand(no_of_particles)
R2 = np.random.rand(no_of_particles)
R1 = af.to_array(R1)
R2 = af.to_array(R2)


# R1 = np.random.rand(no_of_particles)
# R2 = np.random.rand(no_of_particles)
# R3 = np.random.rand(no_of_particles)
# R4 = np.random.rand(no_of_particles)

# Sampling velocities corresponding to Maxwell-Boltzmann distribution at T_initial
# For this we shall be using the Box-Muller transformation
# constant_multiply  = np.sqrt(2 * boltzmann_constant*T_initial/mass_particle)
# initial_velocity_x = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R1)
# initial_velocity_y = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R1)
# initial_velocity_z = constant_multiply*af.arith.sqrt(-af.arith.log(R4))*af.arith.cos(2*np.pi*R3)

constant_multiply  = np.sqrt(2*boltzmann_constant * T_initial/mass_particle)

initial_velocity_x = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R1)


# initial_velocity_x = (af.Array([0.1, 0.3, 0.4, 0.6, 0.95, 0.82])).as_type(af.Dtype.f64)
#constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R1)


# initial_velocity_y = 0 * R1
# initial_velocity_z = 0 * R1

# initial_velocity_x = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.cos(2*np.pi*R1)
# initial_velocity_y = constant_multiply*af.arith.sqrt(-af.arith.log(R2))*af.arith.sin(2*np.pi*R1)
# initial_velocity_z = constant_multiply*af.arith.sqrt(-af.arith.log(R4))*af.arith.cos(2*np.pi*R3)

initial_velocity_y = af.data.constant(0, no_of_particles, dtype = af.Dtype.f64)
initial_velocity_z = af.data.constant(0, no_of_particles, dtype = af.Dtype.f64)

""" Time parameters for the simulation """

# Any time parameter changes that need to be made for the simulation should be edited here:
# box_crossing_time_scale = (length_box_x/np.max(initial_velocity_x))
# final_time              = 3 #5 * box_crossing_time_scale
# dt                      = 0.001#0.1 * box_crossing_time_scale
# time                    = np.arange(0, final_time, dt)

""" Parameters for fields """

# The following lines define the staggered set of points which shall be used in solving of electric and magnetic fields

if(fields_enabled == "true"):
  dx       = length_box_x/x_zones_field
  dy       = length_box_y/y_zones_field

  x_center = np.linspace(-ghost_cells*dx, length_box_x + ghost_cells*dx, x_zones_field + 1 + 2*ghost_cells)
  y_center = np.linspace(-ghost_cells*dy, length_box_y + ghost_cells*dy, y_zones_field + 1 + 2*ghost_cells)

  """ Setting the offset spatial grids """

  x_right = np.linspace(-ghost_cells*(dx) + dx/2, length_box_x + (2*ghost_cells + 1)*(dx/2), x_zones_field + 1 + 2*ghost_cells)
  y_top   = np.linspace(-ghost_cells*(dy) + dy/2, length_box_y + (2*ghost_cells + 1)*(dy/2), y_zones_field + 1 + 2*ghost_cells)

  final_time = 3
  dt         = np.float(dx / (2* 10 * speed_of_light))
  time       = np.arange(0, final_time, dt)

""" Writing the data to a file which can be accessed by a solver"""

h5f = h5py.File('data_files/initial_conditions/initial_data.h5', 'w')
h5f.create_dataset('time',                   data = time)
h5f.create_dataset('x_coords', data = initial_position_x)
h5f.create_dataset('y_coords', data = initial_position_y)
h5f.create_dataset('z_coords', data = initial_position_z)
h5f.create_dataset('vel_x',    data = initial_velocity_x)
h5f.create_dataset('vel_y',    data = initial_velocity_y)
h5f.create_dataset('vel_z',    data = initial_velocity_z)

if(fields_enabled == "true"):
  h5f.create_dataset('x_center',      data = x_center)
  h5f.create_dataset('y_center',      data = y_center)
  h5f.create_dataset('x_right',        data = x_right)
  h5f.create_dataset('y_top',            data = y_top)

h5f.close()
