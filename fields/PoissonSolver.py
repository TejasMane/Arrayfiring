import arrayfire as af
import numpy as np
# from wall_options.EM_periodic import periodic

# Successive over relaxation Poisson solver

def SOR(rho_with_ghost, ghost_cells, dx, dy, *args, **kwargs):
    # print('rho_with_ghost is ', rho_with_ghost)
    # print('ghost_cells is ', ghost_cells)
    # print('dx is ', dx)
    # print('dy is ', dy)
    print('Inside Poisson')
    x_points = (rho_with_ghost[0, :]).elements()
    y_points = (rho_with_ghost[:, 0]).elements()
    # rho_with_ghost = periodic(rho_with_ghost, y_points, x_points, ghost_cells)

    omega   = kwargs.get('omega', None)
    epsilon = kwargs.get('epsilon', None)
    max_iterations = kwargs.get('max_iterations', None)

    l = x_points
    if(omega == None):
        omega = 2/(1+(np.pi/l)) - 1

    if(epsilon == None):
        epsilon = 1e-4

    if(max_iterations == None):
        max_iterations = 1500

    # print('omega is ', omega)

    X_physical_index = ghost_cells + af.data.range(y_points - 2 * ghost_cells, d1= x_points - 2 * ghost_cells, dim=1)
    Y_physical_index = ghost_cells + af.data.range(y_points - 2 * ghost_cells, d1= x_points - 2 * ghost_cells, dim=0)


    V_k      = af.data.constant(0, (rho_with_ghost[:, 0]).elements(), (rho_with_ghost[0, :]).elements(), dtype=af.Dtype.f64)
    V_k[ 0, :] = 1
    V_k[-1, :] = 2
    V_k_plus = af.data.constant(0, (rho_with_ghost[:, 0]).elements(), (rho_with_ghost[0, :]).elements(), dtype=af.Dtype.f64)
    V_k_plus[ 0, :] = 1
    V_k_plus[-1, :] = 2

    # omega = 0.6

    V_k_plus =  (1-omega) * V_k \
                                                    + (omega/(2*dx**2 + 2*dy**2)) \
                                                    * (    (dy**2)*(af.data.shift(V_k,0,1) + af.data.shift(V_k,0,-1)) \
                                                        + (dx**2)*(af.data.shift(V_k,1,0) + af.data.shift(V_k,-1,0)) \
                                                        + ((dx**2)*dy**2)*(rho_with_ghost) \
                                                      )

    V_k_plus[ 0, :] = 1
    V_k_plus[-1, :] = 2
    V_k_plus[ :, 0] = 0
    V_k_plus[:, -1] = 0
    # print('Before loop Vk plus ', V_k_plus)
    # print('Before loop af.max(af.abs(V_k_plus - V_k)) ',af.max(af.abs(V_k_plus - V_k)) )
    for i in range(max_iterations):
        # print(' iteration')
        V_k = V_k_plus.copy()

        V_k_plus =  (1-omega) * V_k \
                                                        + (omega/(2*dx**2 + 2*dy**2)) \
                                                        * (    (dy**2)*(af.data.shift(V_k,0,1) + af.data.shift(V_k,0,-1)) \
                                                            + (dx**2)*(af.data.shift(V_k,1,0) + af.data.shift(V_k,-1,0)) \
                                                            + ((dx**2)*dy**2)*(rho_with_ghost) \
                                                          )

        V_k_plus[ 0, :] = 1
        V_k_plus[-1, :] = 2
        V_k_plus[ :, 0] = 0
        V_k_plus[:, -1] = 0


        if(i%10 == 0):
            if(i%100==0):
                print('Inside Poisson')
            if(af.max(af.abs(V_k_plus - V_k)) < epsilon):
                return V_k_plus
        # print('Vk is ', V_k)
        # print('V_k update = ',V_k_plus)
        # print('Convergence =',af.max(af.abs(V_k_plus - V_k)))
        # print('Vk plus ', V_k_plus)
        # input('check')


    af.eval(V_k_plus)
    return V_k_plus
