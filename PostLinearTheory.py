import numpy as np
import h5py
import params
import pylab as pl
import arrayfire as af
import initialize

pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

no_of_particles = params.no_of_particles
length_box_x = params.length_box_x
x_divisions = initialize.x_divisions_perturbed
left_boundary = params.left_boundary
right_boundary = params.right_boundary
ghost_cells = params.ghost_cells

h5f           = h5py.File('data_files/initial_conditions/initial_data.h5', 'r')

x_initial     = h5f['x_coords'][:]

y_initial     = h5f['y_coords'][:]
y_initial     = (af.to_array(y_initial)).as_type(af.Dtype.f64)

vel_x_initial = h5f['vel_x'][:]
vel_x_initial = af.to_array(0.2*vel_x_initial)

vel_y_initial = h5f['vel_y'][:]
vel_y_initial = af.to_array(0.2*vel_y_initial)

time          = h5f['time'][:]
print('time length', time.size)
x_center      = h5f['x_center'][:]
x_center      = af.to_array(x_center)

y_center      = h5f['y_center'][:]
y_center      = af.to_array(y_center)

x_right       = h5f['x_right'][:]
x_right        = af.to_array(x_center)

y_top         = h5f['y_top'][:]
y_top         = af.to_array(y_center)

z_initial     = h5f['z_coords'][:]
z_initial     = af.to_array(z_initial)

vel_z_initial = h5f['vel_z'][:]
vel_z_initial     = af.to_array(0.2*vel_z_initial)

h5f.close()
x_temp = np.linspace(0,1,100)
a, b = np.histogram(x_initial, bins=(x_divisions), range=(left_boundary, right_boundary))
a = (a / (no_of_particles / x_divisions))-1
pl.plot(x_temp,a)
pl.xlabel('$x$')
pl.ylabel(r'$\delta\rho(x)$')
pl.show()
pl.clf()

data = np.zeros((time.size), dtype = np.float)

for time_index,t0 in enumerate(time):
    if(time_index%100 ==0):
        print(time_index)
    if(time_index == time.size -1):
        break
    # if (time_index % 10 == 0):
    h5f = h5py.File('data_files/timestepped_data/solution_'+str(time_index)+'.h5', 'r')
    x_coords = h5f['x_coords'][:]
    h5f.close()

    a,b = np.histogram(x_coords, bins=(x_divisions), range=(left_boundary, right_boundary) )
    data[time_index] = np.log(max(abs((a/(no_of_particles/x_divisions))-1)))
    # print('time = ', time_index, 'Amplitude = ', max(a))
    # pl.plot(a)
    # pl.savefig('data_files/images/' + '%04d' % (time_index) + '.png')
    # pl.clf


pl.plot(time,data)
pl.xlabel('$t$')
pl.ylabel(r'$\delta\rho(t)$')
pl.show()
pl.clf()