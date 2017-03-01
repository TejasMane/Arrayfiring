import numpy as np
from wall_options.EM_periodic import periodic
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
forward_row        = params.forward_row
forward_column     = params.forward_column
backward_row       = params.backward_row
backward_column    = params.backward_column
identity           = params.identity


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


""" Equations for mode 1 FDTD"""

# dEz/dt = dBy/dx - dBx/dy
# dBx/dt = -dEz/dy
# dBy/dt = +dEz/dx
# div_B  = dBx/dx + dBy/dy

""" Equations for mode 2 FDTD"""

# dBz/dt = - ( dEy/dx - dEx/dy )
# dEx/dt = + dBz/dy
# dEy/dt = - dBz/dx
# div_B  = dBz/dz

"""
Notes for periodic boundary conditions:
for [0, Lx] domain use periodic BC's such that last point in the physical domain coincides with the first point
for [0, Lx) domain use periodic BC's such that the ghost point after the last physical point coincides with the first
physical point
"""


""" Alignment of the spatial grids for the fields(Convention chosen)

# This is the convention which will be used in the matrix representation

positive y axis -------------> going down
positive x axis -------------> going right

Let the domain be [0,1]
Sample grid with one ghost cell at each end and the physical domain containing only 2 points
Here dx = 1, dx/2 = 0.5

Let the grids for the example case be denoted be:

x_center = [-1, 0, 1, 2]
y_center = [-1, 0, 1, 2]

x_center[0] and x_center[3] are the ghost points and x_center[1] and x_center[2] are the physical points
y_center[0] and y_center[3] are the ghost points and y_center[1] and y_center[2] are the physical points


x_right  = [-0.5, 0.5, 1.5, 2.5]
y_top    = [-0.5, 0.5, 1.5, 2.5]

x_right[0] and x_right[3] are the ghost points and x_right[1] and x_right[2] are the physical points
y_top[0] and y_top[3] are the ghost points and y_top[1] and y_top[2] are the physical points

This can be seen visually with the below presented schematic

where pij are the points located on the fused spatial grids for whole numbers i an j

p11, p12, p13, p14, p15, p16, p17, p18, p28, p38, p48, p58, p68, p78, p88, p87, p86, p85, p84, p83, p82,
p81, p71, p61, p51, p41, p31 and p21 are all ghost points while all other points are the physical points for this
example taken.

+++++++++p11--------p12--------p13--------p14--------p15--------p16--------p17--------p18+++++++++++++++++++++++++++++++
          |                                                                            |
          |   p11 = (x_center[0], y_center[0]), p13 = (x_center[1], y_center[0])       |
          |   p15 = (x_center[2], y_center[0]),p17 = (x_center[3], y_center[0])        |
          |   p12 = (x_right[0], y_center[0]), p14 = (x_right[1], y_center[0])         |
          |   p16 = (x_right[2], y_center[0]), p18 = (x_right[3], y_center[0])         |
          |                                                                            |
+++++++++p21--------p22--------p23--------p24--------p25--------p26--------p27--------p28+++++++++++++++++++++++++++++++
          |                                                                            |
          |   p21 = (x_center[0], y_top[0]), p23 = (x_center[1], y_top[0])             |
          |   p25 = (x_center[2], y_top[0]), p27 = (x_center[3], y_top[0])             |
          |   p22 = (x_right[0], y_top[0]), p24 = (x_right[1], y_top[0])               |
          |   p26 = (x_right[2], y_top[0]), p28 = (x_right[3], y_top[0])               |
          |                                                                            |
+++++++++p31--------p32--------p33--------p34--------p35--------p36--------p37--------p38+++++++++++++++++++++++++++++++
          |                                                                            |
          |   p31 = (x_center[0], y_center[1]), p33 = (x_center[1], y_center[1])       |
          |   p35 = (x_center[2], y_center[1]), p37 = (x_center[3], y_center[1])       |
          |   p32 = (x_right[0], y_center[1]), p34 = (x_right[1], y_center[1])         |
          |   p36 = (x_right[2], y_center[1]), p38 = (x_right[3], y_center[1])         |
          |                                                                            |
+++++++++p41--------p42--------p43--------p44--------p45--------p46--------p47--------p48+++++++++++++++++++++++++++++++
          |                                                                            |
          |   p41 = (x_center[0], y_top[1]), p43 = (x_center[1], y_top[1])             |
          |   p45 = (x_center[2], y_top[1]), p47 = (x_center[3], y_top[1])             |
          |   p42 = (x_right[0], y_top[1]), p44 = (x_right[1], y_top[1])               |
          |   p46 = (x_right[2], y_top[1]), p48 = (x_right[3], y_top[1])               |
          |                                                                            |
+++++++++p51--------p52--------p53--------p54--------p55--------p56--------p57--------p58+++++++++++++++++++++++++++++++
          |                                                                            |
          |                                                                            |
          |                                                                            |
          | And So on ................                                                 |
          |                                                                            |
          |                                                                            |
+++++++++p61--------p62--------p63--------p64--------p65--------p66--------p67--------p68+++++++++++++++++++++++++++++++
          |                                                                            |
          |                                                                            |
          | And So on ................                                                 |
          |                                                                            |
          |                                                                            |
          |                                                                            |
+++++++++p71--------p72--------p73--------p74--------p75--------p76--------p77--------p78+++++++++++++++++++++++++++++++
          |                                                                            |
          |                                                                            |
          |                                                                            |
          | And So on ................                                                 |
          |                                                                            |
          |                                                                            |
+++++++++p81--------p82--------p83--------p84--------p85--------p86--------p87--------p88+++++++++++++++++++++++++++++++

Now the fields aligned in x and y direction along with the following grids:


Ez  = (x_center, y_center ) 0, dt, 2dt, 3dt...
Bx  = (x_center, y_top    ) -0.5dt, 0.5dt, 1.5dt, 2.5dt...
By  = (x_right, y_center  ) -0.5dt, 0.5dt, 1.5dt, 2.5dt...

Ex  = (x_right, y_center  ) 0, dt, 2dt, 3dt...
Ey  = (x_center, y_top    ) 0, dt, 2dt, 3dt...
Bz  = (x_right, y_top     ) -0.5dt, 0.5dt, 1.5dt, 2.5dt...

rho = (x_center, y_top    )  # Not needed here

Jx  = (x_right, y_center  ) 0.5dt, 1.5dt, 2.5dt...
Jy  = (x_center, y_top    ) 0.5dt, 1.5dt, 2.5dt...
Jz  = (x_center, y_center ) 0.5dt, 1.5dt, 2.5dt...
"""


""" Equations for mode 1 fdtd (variation along x and y)"""

# dEz/dt = dBy/dx - dBx/dy
# dBx/dt = -dEz/dy
# dBy/dt = +dEz/dx
# div_B = dBx/dx + dBy/dy

def mode1_fdtd( Ez, Bx, By, Lx, Ly, c, ghost_cells, Jx, Jy, Jz, dt,no_of_particles):

  """ Number of grid points in the field's domain"""

  (x_number_of_points,  y_number_of_points) = Ez.dims()

  """ number of grid zones from the input fields """

  Nx = x_number_of_points - 2*ghost_cells - 1
  Ny = y_number_of_points - 2*ghost_cells - 1

  """ local variables for storing the input fields """

  Ez_local = Ez
  Bx_local = Bx
  By_local = By

  """Enforcing BC's"""

  Ez_local = periodic(Ez_local, y_number_of_points, x_number_of_points, ghost_cells)

  Bx_local = periodic(Bx_local, y_number_of_points, x_number_of_points, ghost_cells)

  By_local = periodic(By_local, y_number_of_points, x_number_of_points, ghost_cells)

  """ Setting division size and time steps"""

  dx = np.float(Lx / (Nx))
  dy = np.float(Ly / (Ny))

  """ defining variables for convenience """

  dt_by_dx = dt / (dx)
  dt_by_dy = dt / (dy)


  """  Updating the Magnetic fields   """

  Bx_local += -dt_by_dy*(af.signal.convolve2_separable(forward_row, identity, Ez_local))

  # dBx/dt = -dEz/dy

  By_local += dt_by_dx*(af.signal.convolve2_separable(identity, forward_column, Ez_local))

  # dBy/dt = +dEz/dx

  """  Implementing periodic boundary conditions using ghost cells  """

  Bx_local = periodic(Bx_local, y_number_of_points, x_number_of_points, ghost_cells)

  By_local = periodic(By_local, y_number_of_points, x_number_of_points, ghost_cells)


  """  Updating the Electric field using the current too """

  Ez_local +=   dt_by_dx * (af.signal.convolve2_separable(identity, backward_column, By_local)) \
              - dt_by_dy * (af.signal.convolve2_separable(backward_row, identity, Bx_local)) \
              - dt*(Jz/no_of_particles)

  # dEz/dt = dBy/dx - dBx/dy

  """  Implementing periodic boundary conditions using ghost cells  """

  Ez_local = periodic(Ez_local, y_number_of_points, x_number_of_points, ghost_cells)

  af.eval(Ez_local, Bx_local, By_local)
  return Ez_local, Bx_local, By_local



"""-------------------------------------------------End--of--Mode--1-------------------------------------------------"""


"""-------------------------------------------------Start--of--Mode-2------------------------------------------------"""

""" Equations for mode 2 fdtd (variation along x and y)"""

# dBz/dt = - ( dEy/dx - dEx/dy )
# dEx/dt = + dBz/dy
# dEy/dt = - dBz/dx
# div_B = dBz/dz


def mode2_fdtd( Bz, Ex, Ey, Lx, Ly, c, ghost_cells, Jx, Jy, Jz, dt):

  """ Number of grid points in the field's domain """

  (x_number_of_points,  y_number_of_points) = Bz.dims()

  """ number of grid zones calculated from the input fields """

  Nx = x_number_of_points - 2*ghost_cells-1
  Ny = y_number_of_points - 2*ghost_cells-1

  """ local variables for storing the input fields """

  Bz_local = Bz
  Ex_local = Ex
  Ey_local = Ey

  """Enforcing periodic BC's"""

  Bz_local = periodic(Bz_local, y_number_of_points, x_number_of_points, ghost_cells)

  Ex_local = periodic(Ex_local, y_number_of_points, x_number_of_points, ghost_cells)

  Ey_local = periodic(Ey_local, y_number_of_points, x_number_of_points, ghost_cells)


  """ Setting division size and time steps"""

  dx = np.float(Lx / (Nx))
  dy = np.float(Ly / (Ny))

  """ defining variable for convenience """

  dt_by_dx = dt / (dx)
  dt_by_dy = dt / (dy)



  """  Updating the Magnetic field  """

  Bz_local += - dt_by_dx * (af.signal.convolve2_separable(identity, forward_column, Ey_local)) \
              + dt_by_dy * (af.signal.convolve2_separable(forward_row, identity, Ex_local))

  # dBz/dt = - ( dEy/dx - dEx/dy )

  #Implementing periodic boundary conditions using ghost cells

  Bz_local = periodic(Bz_local, y_number_of_points, x_number_of_points, ghost_cells)


  """  Updating the Electric fields using the current too   """

  Ex_local += dt_by_dy * (af.signal.convolve2_separable(backward_row, identity, Bz_local)) - (Jx/no_of_particles) * dt

  # dEx/dt = + dBz/dy

  Ey_local += -dt_by_dx * (af.signal.convolve2_separable(identity, backward_column, Bz_local)) - (Jy/no_of_particles) * dt

  # dEy/dt = - dBz/dx

  """  Implementing periodic boundary conditions using ghost cells  """

  Ex_local = periodic(Ex_local, y_number_of_points, x_number_of_points, ghost_cells)

  Ey_local = periodic(Ey_local, y_number_of_points, x_number_of_points, ghost_cells)



  af.eval(Bz_local, Ex_local, Ey_local)

  return Bz_local, Ex_local, Ey_local

"""-------------------------------------------------End--of--Mode--2-------------------------------------------------"""

def fdtd(Ex, Ey, Ez, Bx, By, Bz, c, Lx, Ly, ghost_cells, Jx, Jy, Jz, dt):

  # Decoupling the fields to solve for them individually

  Ez_updated, Bx_updated, By_updated = mode1_fdtd(Ez, Bx, By, Lx, Ly, c, ghost_cells, Jx, Jy, Jz, dt)

  Bz_updated, Ex_updated, Ey_updated = mode2_fdtd(Bz, Ex, Ey, Lx, Ly, c, ghost_cells, Jx, Jy, Jz, dt)
  af.eval(Ex_updated, Ey_updated, Ez_updated, Bx_updated, By_updated, Bz_updated)

  # combining the the results from both modes

  af.eval(Ex_updated, Ey_updated, Ez_updated, Bx_updated, By_updated, Bz_updated)

  return Ex_updated, Ey_updated, Ez_updated, Bx_updated, By_updated, Bz_updated
