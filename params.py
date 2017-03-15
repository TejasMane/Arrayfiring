# Stability criterion
# dx < 3 * (root(T/ne^2))

no_of_particles      = 100000
mass_particle        = 1.0
boltzmann_constant   = 1.0

# If the user requests the written data files will contain the spacial temperature array of each time-step
plot_spatial_temperature_profile = "false"

if(plot_spatial_temperature_profile == "true"):
  x_zones_temperature = 100
  y_zones_temperature = 100

"""Defining the choice for the integrator"""
# The following options are currently available for integrators:
# Option "verlet"    - Velocity Verlet algorithm

choice_integrator = "verlet"

""" Defining the collision parameters """
# The following collision kernels are available for implementation
# Option "collisionless"   - Collisionless Model
# Option "potential-based" - Potential Based Scattering
# Option "montecarlo"      - MonteCarlo Scattering

collision_operator = "collisionless"

# The minimum scattering distance can only be altered by shifting the potential function found in collision_operators/potential.py
# Default function for potential = potential_magnitude * (-tanh(potential_gradient*distance) + 1)
# Here you can change the potential gradient and potential magnitude, and the order used for finite differencing
if(collision_operator == "potential-based"):
  potential_steepness     = 300
  potential_amplitude     = 20
  order_finite_difference = 4

# in Monte-Carlo scattering, particles are scattered depending upon the local temperature of the zone they prevail in
# this eliminates the distance check between particles.
# Although a selected fraction of the particles can be scattered. This hasn't been implemented yet
# Currently all particles in the zone are scattered
elif(collision_operator == "montecarlo"):
  x_zones_montecarlo = 50
  y_zones_montecarlo = 50

""" Definining parameters for electric and magnetic fields """

fields_enabled   = "true"

import arrayfire as af
if(fields_enabled == "true"):
  spread          = 0.1     # Shall be used to assign Gaussian
  ghost_cells     = 1       # Refers to the number of cells beyond the physical domain(usually set to 1)
  speed_of_light  = 1
  charge          = -1      # Charge of each individual particle in the simulation
  x_zones_field   = 200    # Refers to the number of x-divisions for the cells that are used to compute fields, and currents
  y_zones_field   = 4      # Refers to the number of y-divisions for the cells that are used to compute fields, and currents
  forward_row     = af.Array([1, -1, 0])
  forward_column  = af.Array([1, -1, 0])
  backward_row    = af.Array([0, 1, -1])
  backward_column = af.Array([0, 1, -1])
  identity        = af.Array([0, 1, 0] )

"""
[[1, -1, 0]] = data[col +1] - data[col]

[[1], [-1], [0]] = data[row]=  data[row + 1] - data[row]

[[0], [1], [-1]] = data[row] = data[row] - data[row-1]

[[0, 1, -1]] = data[col] = data[col] - data[col-1]

"""



""" Wall and temperature parameters """

# We shall define the different wall conditions that may be implemented:
# Option "periodic" - Periodic B.C's at the walls
# Option "hardwall" - Hardwall B.C's at the walls
# Option "thermal"  - Thermal B.C's at the walls

T_initial        = 1.0

wall_condition_x = "periodic"
wall_condition_y = "periodic"
wall_condition_z = "periodic"

if(wall_condition_x == "thermal"):
  T_left_wall  = 2.0
  T_right_wall = 2.0

if(wall_condition_y == "thermal"):
  T_top_wall = 2.0
  T_bot_wall = 2.0

if(wall_condition_z == "thermal"):
  T_front_wall = 2.0
  T_back_wall  = 2.0

import numpy as np
""" Length Parameters of Simulation Domain """

left_boundary    = 0.
right_boundary   = 1.
length_box_x     = right_boundary - left_boundary

bottom_boundary  = 0.
top_boundary     = 1.
length_box_y     = top_boundary   - bottom_boundary

back_boundary    = 0.
front_boundary   = 1.
length_box_z     = front_boundary - back_boundary

""" Linear Theory terms"""

k_fourier = 2*np.pi
Amplitude_perturbed = 0.1
