import arrayfire as af
import numpy as np
from fields.current_depositor import Umeda_2003

user_defined_test = 1

def run_test(x):
    if(x==1):
        test_1()

    if(x==2):
        test_2()

    if(x==3):
        test_3()


    return 1


def test_1():
    charge = 1

    no_of_particles = 2
    # no_of_particles = 1


    positions_x = af.Array([0.7, 0.3])
    positions_y = af.Array([0.7, 0.3])
    positions_z = af.Array([0.5, 0.5])

    velocities_x = af.Array([0.2, 0.2])
    velocities_y = af.Array([0.2, 0.2])
    velocities_z = af.Array([0.0, 0.0])

    # positions_x = af.Array([0.7])
    # positions_y = af.Array([0.7])
    # positions_z = af.Array([0.5])
    #
    # velocities_x = af.Array([0.2])
    # velocities_y = af.Array([0.2])
    # velocities_z = af.Array([0.0])

    x_center_grid = af.Array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    y_center_grid = af.Array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

    dt = 0.2

    dx = 0.2
    dy = 0.2

    ghost_cells = 1

    Lx = 1.0
    Ly = 1.0

    Jx, Jy, Jz = Umeda_2003( charge, no_of_particles, positions_x ,positions_y,\
                             positions_z, velocities_x, velocities_y, velocities_z, \
                             x_center_grid, y_center_grid, ghost_cells,\
                             Lx, Ly, dx, dy, dt\
                           )

    print('Jx is ', Jx)
    print('Jy is ', Jy)
    print('Jz is ', Jz)



    return 1



run_test(user_defined_test)
