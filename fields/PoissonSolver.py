import arrayfire as af
import numpy as np
from wall_options.EM_periodic import periodic

# Successive over relaxation Poisson solver

def SOR(rho_with_ghost, ghost_cells, dx, dy, *args, **kwargs):

    x_points = (rho_with_ghost[0, :]).elements()
    y_points = (rho_with_ghost[:, 0]).elements()
    rho_with_ghost = periodic(rho_with_ghost, y_points, x_points, ghost_cells)

    omega   = kwargs.get('omega', None)
    epsilon = kwargs.get('epsilon', None)

    if(omega == None):
        omega = 2/(1+(np.pi/l))

    if(epsilon == None):
        epsilon = 1e-4

    X_physical_index = ghost_cells + af.data.range(y_points - 2 * ghost_cells, d1= x_points - 2 * ghost_cells, dim=1)
    Y_physical_index = ghost_cells + af.data.range(y_points - 2 * ghost_cells, d1= x_points - 2 * ghost_cells, dim=0)


    V_k      = af.data.constant(0, (rho_with_ghost[:, 0]).elements(), (rho_with_ghost[0, :]).elements(), dtype=af.Dtype.f64)
    V_k_plus = af.data.constant(0, (rho_with_ghost[:, 0]).elements(), (rho_with_ghost[0, :]).elements(), dtype=af.Dtype.f64)

    while(af.sum(af.abs(V_k_plus - V_k))<epsilon):
        V_k = V_k_plus.copy()


        V_k_plus[X_physical_index, Y_physical_index] =  (1-omega) * V_k[X_physical_index, Y_physical_index] \
                                                        + (omega/(( 1/dx**2 )+(1/ dy**2))) \
                                                        * (    (1/dx**2)*(V_k[X_physical_index, Y_physical_index + 1] + V_k[X_physical_index, Y_physical_index - 1]) \
                                                            + (1/dy**2)*(V_k[X_physical_index + 1, Y_physical_index] + V_k[X_physical_index - 1, Y_physical_index]) \
                                                            - (rho_with_ghost[X_physical_index, Y_physical_index]) \
                                                          )

    return V_k_plus
