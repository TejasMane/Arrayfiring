from fields.PoissonSolver import SOR
import numpy as np
import pylab as pl
import arrayfire as af
import h5py



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

ghost_cells = 1

x_divisions_physical = 80
y_divisions_physical = 80

dx = 1/x_divisions_physical
dy = 1/y_divisions_physical

x_points = x_divisions_physical + 2 * ghost_cells + 1
y_points = y_divisions_physical + 2 * ghost_cells + 1

rho = af.data.constant(0, x_points, y_points, dtype=af.Dtype.f64)

poisson_solution  = SOR(rho, ghost_cells, dx, dy)

h5f = h5py.File('data_files/poisson_solution.h5', 'w')
h5f.create_dataset('poisson_solution',   data = poisson_solution)
h5f.close()

h5f = h5py.File('data_files/poisson_solution.h5', 'r')
poisson_solution = h5f['poisson_solution'][:]
h5f.close()

x = np.linspace(0,1,x_points)
y = np.linspace(0,1,y_points)

pl.contourf(poisson_solution,100,cmap = 'jet')
pl.colorbar()
pl.xlabel('$x$')
pl.ylabel('$y$')
pl.title('$V$')
pl.show()
pl.clf()
