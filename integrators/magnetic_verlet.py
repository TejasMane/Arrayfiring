import arrayfire as af

"""
This is the integrator that implements the Boris Algorithm.
Add details here

Input positions  :  x(n)
Input Velocities :  v(n+0.5dt)
Input fields : E(x((n + 1)*dt)), B(x((n+1)*dt))

"""

def integrator(mass_particle, charge, x_coords, y_coords, z_coords, vel_x, vel_y, vel_z, dt, Ex, Ey, Ez, Bx, By, Bz):

  vel_x_minus = vel_x + (charge * Ex * dt) / (2 * mass_particle)
  vel_y_minus = vel_y + (charge * Ey * dt) / (2 * mass_particle)
  vel_z_minus = vel_z + (charge * Ez * dt) / (2 * mass_particle)

  t_magx    = (charge * Bx * dt) / (2 * mass_particle)
  t_magy    = (charge * By * dt) / (2 * mass_particle)
  t_magz    = (charge * Bz * dt) / (2 * mass_particle)

  vminus_cross_t_x =  (vel_y_minus * t_magz) - (vel_z_minus * t_magy)
  vminus_cross_t_y = -(vel_x_minus * t_magz) + (vel_z_minus * t_magx)
  vminus_cross_t_z =  (vel_x_minus * t_magy) - (vel_y_minus * t_magx)

  vel_dashx = vel_x_minus + vminus_cross_t_x
  vel_dashy = vel_y_minus + vminus_cross_t_y
  vel_dashz = vel_z_minus + vminus_cross_t_z

  t_mag = af.arith.sqrt(t_magx ** 2 + t_magy ** 2 + t_magz ** 2)

  s_x = (2 * t_magx) / (1 + af.arith.abs(t_mag ** 2))
  s_y = (2 * t_magy) / (1 + af.arith.abs(t_mag ** 2))
  s_z = (2 * t_magz) / (1 + af.arith.abs(t_mag ** 2))

  vel_x_plus = vel_x_minus + ((vel_dashy * s_z) - (vel_dashz * s_y))
  vel_y_plus = vel_y_minus - ((vel_dashx * s_z) - (vel_dashz * s_x))
  vel_z_plus = vel_z_minus + ((vel_dashx * s_y) - (vel_dashy * s_x))

  vel_x_new  = vel_x_plus + (charge * Ex * dt) / (2 * mass_particle)
  vel_y_new  = vel_y_plus + (charge * Ey * dt) / (2 * mass_particle)
  vel_z_new  = vel_z_plus + (charge * Ez * dt) / (2 * mass_particle)

  # Using v at (n+0.5) dt to push x at (n)dt

  x_coords_new = x_coords + vel_x * dt
  y_coords_new = y_coords + vel_y * dt
  z_coords_new = z_coords + vel_z * dt

  af.eval(x_coords_new, y_coords_new, z_coords_new)
  af.eval(vel_x_new, vel_y_new, vel_z_new)

  return (x_coords_new, y_coords_new, z_coords_new,\
          vel_x_new   , vel_y_new   , vel_z_new\
         )
