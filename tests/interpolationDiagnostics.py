import numpy as np
import h5py
import pylab as pl
import arrayfire as af
from fields.interpolator import fraction_finder


pl.rcParams['figure.figsize'] = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family'] = 'serif'
pl.rcParams['font.weight'] = 'bold'
pl.rcParams['font.size'] = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex'] = True
pl.rcParams['axes.linewidth'] = 1.5
pl.rcParams['axes.titlesize'] = 'medium'
pl.rcParams['axes.labelsize'] = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad'] = 8
pl.rcParams['xtick.minor.pad'] = 8
pl.rcParams['xtick.color'] = 'k'
pl.rcParams['xtick.labelsize'] = 'medium'
pl.rcParams['xtick.direction'] = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad'] = 8
pl.rcParams['ytick.minor.pad'] = 8
pl.rcParams['ytick.color'] = 'k'
pl.rcParams['ytick.labelsize'] = 'medium'
pl.rcParams['ytick.direction'] = 'in'


def sumsum(a):
    return af.sum(af.abs(a))


def initial_fields(x, y):
    function_value = af.arith.sin(2 * np.pi * x * y) * af.arith.cos(2 * np.pi * x * y)

    return function_value


# Now we shall proceed to evolve the system with time:

""" Error function """


def interpolation_error_convergence(a, b):
    Ez_error = af.data.constant(0, a.elements(), dtype=af.Dtype.f64)
    Bx_error = af.data.constant(0, a.elements(), dtype=af.Dtype.f64)
    By_error = af.data.constant(0, a.elements(), dtype=af.Dtype.f64)

    for outer_index in range(a.elements()):

        print('Computing error for Nx = ', af.sum(a[outer_index]))

        """ Getting the two dimension matrix for initializing the fields """

        Nx = af.sum(a[outer_index])  # number of zones not points
        Ny = af.sum(b[outer_index])  # number of zones not points

        """ Length of each zone along x and y """

        length_box_x = 1
        length_box_y = 1

        speed_of_light = 1
        ghost_cells = 1

        dx = np.float(length_box_x / (Nx))
        dy = np.float(length_box_y / (Ny))

        """ Initializing the spatial grids"""

        # np.linspace(start point, endpoint, number of points, endpoint = Tue/False)

        x_center = np.linspace(-ghost_cells * dx, length_box_x + ghost_cells * dx, Nx + 1 + 2 * ghost_cells,
                               endpoint=True)
        y_center = np.linspace(-ghost_cells * dy, length_box_y + ghost_cells * dy, Ny + 1 + 2 * ghost_cells,
                               endpoint=True)

        x_right = np.linspace(-ghost_cells * dx / 2, length_box_x + (2 * ghost_cells + 1) * dx / 2,
                              Nx + 1 + 2 * ghost_cells, \
                              endpoint=True \
                              )

        y_top = np.linspace(-ghost_cells * dy / 2, length_box_y + (2 * ghost_cells + 1) * dy / 2,
                            Ny + 1 + 2 * ghost_cells, \
                            endpoint=True \
                            )

        x_center = af.to_array(x_center)
        y_center = af.to_array(y_center)
        x_right = af.to_array(x_right)
        y_top = af.to_array(y_top)

        X_center_physical = af.tile(af.reorder(x_center[ghost_cells:-ghost_cells], 1),
                                    y_center[ghost_cells:-ghost_cells].elements(), 1)

        X_right_physical = af.tile(af.reorder(x_right[ghost_cells:-ghost_cells], 1),
                                   y_center[ghost_cells:-ghost_cells].elements(), 1)

        Y_center_physical = af.tile(y_center[ghost_cells:-ghost_cells], 1,
                                    x_center[ghost_cells:-ghost_cells].elements())

        Y_top_physical = af.tile(y_top[ghost_cells:-ghost_cells], 1, x_center[ghost_cells:-ghost_cells].elements())

        """ Initializing the field variables """

        Ez = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
        Bx = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)
        By = af.data.constant(0, x_center.elements(), y_center.elements(), dtype=af.Dtype.f64)

        """ [-ghostcells:ghostcells] selects the points located in the physical domain excluding the ghost cells """

        """ Assigning Field values to the physical physical domain """
        # You can change the initialization here but to get the correct convergence plots make sure error is
        # computed correctly

        Ez[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = initial_fields(X_center_physical, \
                                                                                Y_center_physical
                                                                                )

        Bx[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = initial_fields(X_center_physical,   Y_top_physical )

        By[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells] = initial_fields(X_right_physical, \
                                                                                Y_center_physical \
                                                                                )

        """ Implementing Periodic Boundary conditions using ghost cells """

        from wall_options.EM_periodic import periodic

        Ez = periodic(Ez, (x_center.elements()), (y_center.elements()), ghost_cells)
        Bx = periodic(Bx, (x_center.elements()), (y_top.elements()), ghost_cells)
        By = periodic(By, (x_right.elements()), (y_center.elements()), ghost_cells)

        """ Selecting a number of test points for testing error """

        number_random_points = 50

        x_random = (af.randu(number_random_points)).as_type(af.Dtype.f64)
        y_random = (af.randu(number_random_points)).as_type(af.Dtype.f64)

        """ Selecting a number of test points for testing error """

        Ez_at_random = af.data.constant(0, number_random_points)
        Bx_at_random = af.data.constant(0, number_random_points)
        By_at_random = af.data.constant(0, number_random_points)

        fracs_Ez_x, fracs_Ez_y = fraction_finder(x_random, y_random, x_center, y_center)


        fracs_Bx_x, fracs_Bx_y = fraction_finder(x_random, y_random, x_center, y_top)


        fracs_By_x, fracs_By_y = fraction_finder(x_random, y_random, x_right, y_center)

        """ Calculating interpolated values at the randomly selected points """

        Ez_at_random = af.signal.approx2(Ez, fracs_Ez_y, fracs_Ez_x, method= af.INTERP.LINEAR)

        Bx_at_random = af.signal.approx2(Bx, fracs_Bx_y, fracs_Bx_x, method= af.INTERP.LINEAR)

        By_at_random = af.signal.approx2(By, fracs_By_y, fracs_By_x, method= af.INTERP.LINEAR)

        """ Calculating average errors in the interpolated values at the randomly selected points """

        # Make sure the analytical results at the interpolated points are correct to get 2nd order convergence

        Ez_error[outer_index] = sumsum(Ez_at_random - initial_fields(x_random, y_random)) / number_random_points
        Bx_error[outer_index] = sumsum(Bx_at_random - initial_fields(x_random, y_random)) / number_random_points
        By_error[outer_index] = sumsum(By_at_random - initial_fields(x_random, y_random)) / number_random_points

    return (Ez_error), (Bx_error), (By_error)


""" Choosing test grid densities """

# change N here as desired for the convergence test

# N = np.array([32, 64, 128, 256, 512, 1024])

N = np.arange(100, 3100, 100)
N = af.to_array(N)
# N = af.Array([5])
""" Computing error at the corresponding grid densities """

error_N_Ez, error_N_Bx, error_N_By = interpolation_error_convergence(N, N)

""" Plotting error vs grid density """

# Change this following segment to get plots as desired

pl.loglog(N, error_N_Ez, '-o', lw=3, label='$E_z$ ')
pl.legend()
pl.loglog(N, error_N_Bx, '-o', lw=3, label='$B_x$ ')
pl.legend()
pl.loglog(N, error_N_By, '-o', lw=5, label='$B_y$ ')
pl.legend()
pl.loglog(N, 15 * (N ** -1.999), '--', color='black', lw=2, label=' $O(N^{-2})$ ')
pl.legend()
pl.title('$\mathrm{Convergence\; plot}$ ')
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{L_1\;norm\;of\;error}$')
pl.show()
pl.clf()
