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


def wall_x(x_coords, vel_x, vel_y, vel_z):

  collided_right       = af.algorithm.where(x_coords >= right_boundary)
  collided_left        = af.algorithm.where(x_coords <  left_boundary  )

  if collided_right.elements() > 0:
    x_coords[collided_right] = x_coords[collided_right] - length_box_x

  if collided_left.elements() > 0:
    x_coords[collided_left] = x_coords[collided_left]   + length_box_x

  af.eval(x_coords, vel_x, vel_y, vel_z)

  return x_coords, vel_x, vel_y, vel_z


def wall_y(y_coords, vel_x, vel_y, vel_z):

  collided_top           = af.algorithm.where(y_coords >= top_boundary)
  collided_bottom        = af.algorithm.where(y_coords < bottom_boundary)

  if collided_top.elements()>0:
    y_coords[collided_top] = y_coords[collided_top] - length_box_y

  if collided_bottom.elements()>0:
    y_coords[collided_bottom] = y_coords[collided_bottom] + length_box_y

  af.eval(y_coords, vel_x, vel_y, vel_z)

  return y_coords, vel_x, vel_y, vel_z


def wall_z(z_coords, vel_x, vel_y, vel_z):

  collided_front = af.algorithm.where(z_coords>=front_boundary)
  collided_back  = af.algorithm.where(z_coords<back_boundary  )

  if collided_front.elements()>0:
    z_coords[collided_front] = z_coords[collided_front] - length_box_z

  if collided_back.elements()>0:
    z_coords[collided_back] = z_coords[collided_back]   + length_box_z

  af.eval(z_coords, vel_x, vel_y, vel_z)

  return(z_coords, vel_x, vel_y, vel_z)