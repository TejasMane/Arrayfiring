"""NOT PORTED TO ARRAYFIRE YET"""

import numpy as np
from scipy.special import erfinv
import h5py
import arrayfire as af  
import params

"""Here we shall assign values as set in params"""

no_of_particles      = params.no_of_particles
simulation_dimension = params.simulation_dimension
restart_simulation   = params.restart_simulation
arrayfire_backend    = params.arrayfire_backend
choice_integrator    = params.choice_integrator
collision_operator   = params.collision_operator

if(collision_operator == "hardsphere"):
  scattering_distance = params.scattering_distance

elif(collision_operator == "potential-based"):
  potential_steepness     = params.potential_steepness
  potential_amplitude     = params.potential_amplitude
  order_finite_difference = params.order_finite_difference

elif(collision_operator == "montecarlo"):
  x_zones            = params.x_zones
  y_zones            = params.y_zones
  scattered_fraction = params.scattered_fraction

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
Definition of the integrator:

 Integrator implemented here is the 4-th order Symplectic Integrator developed by Forest & Ruth
 For this integrator the coefficients of the intermediate stages is:
 c_1 = c_4 = 0.67560359598
 c_2 = c_3 = -0.17560359598
 d_1 = d_3 = 1.35120719196
 d_2 = -1.70241438392
 d_4 = 0
"""

def integrator(initial_conditions,dt,potential_steepness = 0):

  x_coordinates = initial_conditions[0:no_of_particles].copy()                     #x^{N}
  y_coordinates = initial_conditions[no_of_particles:2*no_of_particles].copy()     #y^{N}
  velocity_x    = initial_conditions[2*no_of_particles:3*no_of_particles].copy()   #v_x^{N}
  velocity_y    = initial_conditions[3*no_of_particles:4*no_of_particles].copy()   #v_y^{N}

  if(simulation_dimension == 3):

    z_coordinates = initial_conditions[2*no_of_particles:3*no_of_particles].copy()   #z^{N}
    velocity_x    = initial_conditions[3*no_of_particles:4*no_of_particles].copy()   #v_x^{N}
    velocity_y    = initial_conditions[4*no_of_particles:5*no_of_particles].copy()   #v_y^{N}
    velocity_z    = initial_conditions[5*no_of_particles:6*no_of_particles].copy()   #v_z^{N}


  # Manipulating data to implement vectorization

  if(collision_operator == "potential-based"):
    x_coordinates = x_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #x^{N}
    x_coordinates = x_coordinates - np.transpose(x_coordinates)                               #x^{N}

    y_coordinates = y_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #y^{N}
    y_coordinates = y_coordinates - np.transpose(y_coordinates)                               #y^{N}

    if(simulation_dimension == 3):

      z_coordinates = z_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #z^{N}
      z_coordinates = z_coordinates - np.transpose(z_coordinates)                               #z^{N}
      from collision_operators.potential import potential_gradient, potential
      
      distance = np.sqrt(x_coordinates**2+y_coordinates**2) # distance^{N}
      vector   = np.array([x_coordinates, y_coordinates])   # vector^{N}
      nvector  = np.nan_to_num(vector/dist)                 # normalizedvector{N}
      
      force    = potential_gradient(potential_steepness, distance, order_finite_difference) # F^{N}
      force_x  = np.sum(force * nvector[0], axis=1)
      force_y  = np.sum(force * nvector[1], axis=1)
  
    if(simulation_dimension == 3):

      distance = np.sqrt(x_coordinates**2+y_coordinates**2+z_coordinates**2) # distance^{N}
      vector   = np.array([x_coordinates,y_coordinates,z_coordinates])       # vector^{N}
      nvector  = np.nan_to_num(vector/dist)                                  # normalizedvector{N}
      
      force   = potential_gradient(potential_steepness, dist, order_finite_difference) # F^{N}
      force_x = np.sum(force * nvector[0], axis=1)
      force_y = np.sum(force * nvector[1], axis=1)
      force_z = np.sum(force * nvector[2], axis=1)

  if(collision_operator == "potential-based"):

    velocity_x = velocity_x + 1.351207192*(force_x/mass_particle)*dt #v_{x}^{N+1/4}
    velocity_y = velocity_y + 1.351207192*(force_y/mass_particle)*dt #v_{y}^{N+1/4}

    if(simulation_dimension == 3):
    
      velocity_z = velocity_z + 1.351207192*(force_z/mass_particle)*dt #v_{y}^{N+1/4}

    
  x_coordinates = initial_conditions[0:no_of_particles].copy()                 #x^{N}          
  y_coordinates = initial_conditions[no_of_particles:2*no_of_particles].copy() #y^{N}

  x_coordinates_new = x_coordinates + 0.675603596*velocity_x*dt #x^{N+1/4}
  y_coordinates_new = y_coordinates + 0.675603596*velocity_y*dt #y^{N+1/4}

  if(simulation_dimension == 3):
    z_coordinates     = initial_conditions[2*no_of_particles:3*no_of_particles].copy() #z^{N}
    z_coordinates_new = z_coordinates + 0.675603596*velocity_z*dt                      #z^{N+1/4}

  x_coordinates = x_coordinates_new.copy()     #x^{N+1/4}
  y_coordinates = y_coordinates_new.copy()     #y^{N+1/4}

  if(simulation_dimension == 3):

    z_coordinates = z_coordinates_new.copy()   #z^{N+1/4}

  x_coordinates = x_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #x^{N+1/4}
  x_coordinates = x_coordinates - np.transpose(x_coordinates)                               #x^{N+1/4}

  y_coordinates = y_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #y^{N+1/4}
  y_coordinates = y_coordinates - np.transpose(y_coordinates)                               #y^{N+1/4}

  if(simulation_dimension == 3):

    z_coordinates = z_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #z^{N+1/4}
    z_coordinates = z_coordinates - np.transpose(z_coordinates)                               #z^{N+1/4}

  if(collision_operator == "potential-based"):
    
    distance = np.sqrt(x_coordinates**2+y_coordinates**2) # distance^{N+1/4}
    vector   = np.array([x_coordinates, y_coordinates])   # vector^{N+1/4}
    nvector  = np.nan_to_num(vector/dist)                 # normalizedvector{N+1/4}
    
    force    = potential_gradient(potential_steepness, distance, order_finite_difference) # F^{N+1/4}
    force_x  = np.sum(force * nvector[0], axis=1)
    force_y  = np.sum(force * nvector[1], axis=1)
  
    if(simulation_dimension == 3):

      distance = np.sqrt(x_coordinates**2+y_coordinates**2+z_coordinates**2) # distance^{N+1/4}
      vector   = np.array([x_coordinates,y_coordinates,z_coordinates])       # vector^{N+1/4}
      nvector  = np.nan_to_num(vector/dist)                                  # normalizedvector{N+1/4}
      
      force   = potential_gradient(potential_steepness, dist, order_finite_difference) # F^{N+1/4}
      force_x = np.sum(force * nvector[0], axis=1)
      force_y = np.sum(force * nvector[1], axis=1)
      force_z = np.sum(force * nvector[2], axis=1)

  if(collision_operator == "potential-based"):

    velocity_x = velocity_x - 1.702414384*(force_x/mass_particle)*dt #v_{x}^{N+1/2}
    velocity_y = velocity_y - 1.702414384*(force_y/mass_particle)*dt #v_{y}^{N+1/2}

    if(simulation_dimension == 3):
    
      velocity_z = velocity_z - 1.702414384*(force_z/mass_particle)*dt #v_{y}^{N+1/2}

    
  x_coordinates = x_coordinates_new.copy() #x^{N+1/4}          
  y_coordinates = y_coordinates_new.copy() #y^{N+1/4}

  x_coordinates_new = x_coordinates - 0.175603596*velocity_x*dt #x^{N+1/2}
  y_coordinates_new = y_coordinates - 0.175603596*velocity_y*dt #y^{N+1/2}

  if(simulation_dimension == 3):

    z_coordinates     = z_coordinates_new.copy()                  #z^{N+1/4}
    z_coordinates_new = z_coordinates - 0.175603596*velocity_z*dt #z^{N+1/2}
  
  x_coordinates = x_coordinates_new.copy()     #x^{N+1/2}
  y_coordinates = y_coordinates_new.copy()     #y^{N+1/2}

  if(simulation_dimension == 3):

    z_coordinates = z_coordinates_new.copy()   #z^{N+1/4}

  x_coordinates = x_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #x^{N+1/2}
  x_coordinates = x_coordinates - np.transpose(x_coordinates)                               #x^{N+1/2}

  y_coordinates = y_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #y^{N+1/2}
  y_coordinates = y_coordinates - np.transpose(y_coordinates)                               #y^{N+1/2}

  if(simulation_dimension == 3):

    z_coordinates = z_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #z^{N+1/2}
    z_coordinates = z_coordinates - np.transpose(z_coordinates)                               #z^{N+1/2}

  if(collision_operator == "potential-based"):
    
    distance = np.sqrt(x_coordinates**2+y_coordinates**2) # distance^{N+1/2}
    vector   = np.array([x_coordinates, y_coordinates])   # vector^{N+1/2}
    nvector  = np.nan_to_num(vector/dist)                 # normalizedvector{N+1/2}
    
    force    = potential_gradient(potential_steepness, distance, order_finite_difference) # F^{N+1/2}
    force_x  = np.sum(force * nvector[0], axis=1)
    force_y  = np.sum(force * nvector[1], axis=1)
  
    if(simulation_dimension == 3):

      distance = np.sqrt(x_coordinates**2+y_coordinates**2+z_coordinates**2) # distance^{N+1/2}
      vector   = np.array([x_coordinates,y_coordinates,z_coordinates])       # vector^{N+1/2}
      nvector  = np.nan_to_num(vector/dist)                                  # normalizedvector{N+1/2}
      
      force   = potential_gradient(potential_steepness, dist, order_finite_difference) # F^{N+1/2}
      force_x = np.sum(force * nvector[0], axis=1)
      force_y = np.sum(force * nvector[1], axis=1)
      force_z = np.sum(force * nvector[2], axis=1)

  if(collision_operator == "potential-based"):

    velocity_x = velocity_x + 1.351207192*(force_x/mass_particle)*dt #v_{x}^{N+3/4}
    velocity_y = velocity_y + 1.351207192*(force_y/mass_particle)*dt #v_{y}^{N+3/4}

    if(simulation_dimension == 3):
    
      velocity_z = velocity_z + 1.351207192*(force_z/mass_particle)*dt #v_{y}^{N+3/4}

    
  x_coordinates = x_coordinates_new.copy() #x^{N+1/2}          
  y_coordinates = y_coordinates_new.copy() #y^{N+1/2}

  x_coordinates_new = x_coordinates - 0.175603596*velocity_x*dt #x^{N+3/4}
  y_coordinates_new = y_coordinates - 0.175603596*velocity_y*dt #y^{N+3/4}

  if(simulation_dimension == 3):

    z_coordinates     = z_coordinates_new.copy()                  #z^{N+1/2}
    z_coordinates_new = z_coordinates - 0.175603596*velocity_z*dt #z^{N+3/4}

  x_coordinates = x_coordinates_new.copy()     #x^{N+3/4}
  y_coordinates = y_coordinates_new.copy()     #y^{N+3/4}

  if(simulation_dimension == 3):

    z_coordinates = z_coordinates_new.copy()   #z^{N+3/4}

  x_coordinates = x_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #x^{N+3/4}
  x_coordinates = x_coordinates - np.transpose(x_coordinates)                               #x^{N+3/4}

  y_coordinates = y_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #y^{N+3/4}
  y_coordinates = y_coordinates - np.transpose(y_coordinates)                               #y^{N+3/4}

  if(simulation_dimension == 3):

    z_coordinates = z_coordinates * np.ones((no_of_particles,no_of_particles),dtype=np.float) #z^{N+3/4}
    z_coordinates = z_coordinates - np.transpose(z_coordinates)                               #z^{N+3/4}

  if(collision_operator == "potential-based"):
 
    distance = np.sqrt(x_coordinates**2+y_coordinates**2) # distance^{N+3/4}
    vector   = np.array([x_coordinates, y_coordinates])   # vector^{N+3/4}
    nvector  = np.nan_to_num(vector/dist)                 # normalizedvector{N+3/4}
    
    force    = potential_gradient(potential_steepness, distance, order_finite_difference) # F^{N+3/4}
    force_x  = np.sum(force * nvector[0], axis=1)
    force_y  = np.sum(force * nvector[1], axis=1)
  
    if(simulation_dimension == 3):

      distance = np.sqrt(x_coordinates**2+y_coordinates**2+z_coordinates**2) # distance^{N+3/4}
      vector   = np.array([x_coordinates,y_coordinates,z_coordinates])       # vector^{N+3/4}
      nvector  = np.nan_to_num(vector/dist)                                  # normalizedvector{N+3/4}
      
      force   = potential_gradient(potential_steepness, dist, order_finite_difference) # F^{N+3/4}
      force_x = np.sum(force * nvector[0], axis=1)
      force_y = np.sum(force * nvector[1], axis=1)
      force_z = np.sum(force * nvector[2], axis=1)
    
  x_coordinates = x_coordinates_new.copy() #x^{N+3/4}          
  y_coordinates = y_coordinates_new.copy() #y^{N+3/4}

  x_coordinates_new = x_coordinates + 0.675603596*velocity_x*dt #x^{N+1}
  y_coordinates_new = y_coordinates + 0.675603596*velocity_y*dt #y^{N+1}

  if(simulation_dimension == 3):

    z_coordinates     = z_coordinates_new.copy()                  #z^{N+3/4}
    z_coordinates_new = z_coordinates + 0.675603596*velocity_z*dt #z^{N+1}

  if(simulation_dimension == 2):
    next_step = np.concatenate([x_coordinates_new, y_coordinates_new, velocity_x, velocity_y],axis=0)

  else:
    next_step = np.concatenate([x_coordinates_new, y_coordinates_new, z_coordinates_new, velocity_x, velocity_y,velocity_z],axis=0)

  return(next_step)