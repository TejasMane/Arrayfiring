import arrayfire as af
import numpy as np
# from wall_options.EM_periodic import periodic

def Dirichlet(V, ghost_cells):
    # Top wall ghost
    V[ 0, :] = V[-2 - ghost_cells, :]

    # Bottom wall ghost points
    V[-1, :] = V[ghost_cells + 1, :]

    # Left wall ghost
    V[ :, 0] = V[:, -2 - ghost_cells]

    # Right wall ghost
    V[:, -1] = V[:, ghost_cells + 1]

    return V

# Successive over relaxation Poisson solver

def SOR(rho_with_ghost, ghost_cells, dx, dy, *args, **kwargs):

    x_points = (rho_with_ghost[0, :]).elements()
    y_points = (rho_with_ghost[:, 0]).elements()

    omega   = kwargs.get('omega', None)
    epsilon = kwargs.get('epsilon', None)
    max_iterations = kwargs.get('max_iterations', None)

    l = x_points
    if(omega == None):
        omega = 2/(1+(np.pi/l)) - 1

    if(epsilon == None):
        epsilon = 1e-5

    if(max_iterations == None):
        max_iterations = 150000

    X_physical_index = ghost_cells + af.data.range(y_points - 2 * ghost_cells, d1= x_points - 2 * ghost_cells, dim=1)
    Y_physical_index = ghost_cells + af.data.range(y_points - 2 * ghost_cells, d1= x_points - 2 * ghost_cells, dim=0)
    x1 = af.data.range(y_points, dim=0)

    V_k      = af.data.constant(0, (rho_with_ghost[:, 0]).elements(), (rho_with_ghost[0, :]).elements(), dtype=af.Dtype.f64)

    V_k = Dirichlet(V_k, ghost_cells)

    V_k_plus = af.data.constant(0, (rho_with_ghost[:, 0]).elements(), (rho_with_ghost[0, :]).elements(), dtype=af.Dtype.f64)

    V_k_plus = Dirichlet(V_k_plus, ghost_cells)
    # omega = 0.6

    V_k_plus =  (1-omega) * V_k \
                                                    + (omega/(2*dx**2 + 2*dy**2)) \
                                                    * (    (dy**2)*(af.data.shift(V_k,0,1) + af.data.shift(V_k,0,-1)) \
                                                        + (dx**2)*(af.data.shift(V_k,1,0) + af.data.shift(V_k,-1,0)) \
                                                        + ((dx**2)*dy**2)*(rho_with_ghost) \
                                                      )

    V_k_plus = Dirichlet(V_k_plus, ghost_cells)

    for i in range(max_iterations):

        V_k = V_k_plus.copy()

        V_k_plus =  (1-omega) * V_k \
                                                        + (omega/(2*dx**2 + 2*dy**2)) \
                                                        * (    (dy**2)*(af.data.shift(V_k,0,1) + af.data.shift(V_k,0,-1)) \
                                                            + (dx**2)*(af.data.shift(V_k,1,0) + af.data.shift(V_k,-1,0)) \
                                                            + ((dx**2)*dy**2)*(rho_with_ghost) \
                                                          )

        V_k_plus = Dirichlet(V_k_plus, ghost_cells)


        if(i%10 == 0):
            if(i%100==0):
                print('Inside Poisson')
            if(af.max(af.abs(V_k_plus - V_k)) < epsilon):
                print('iteration = ',i)
                # print(V_k_plus)
                return V_k_plus

    af.eval(V_k_plus)
    return V_k_plus


def compute_Electric_field(V, dx, dy):
    # Ex(i, j) = -\nabla\;V = -(V[i,j + 1] - V[i,j])/dx
    # Ey(i,j) = -\nabla\;V = - (V[i + 1, j] - V[i, j])/dy
    # Row column representation

    Ex = -(af.data.shift(V, 0, 1) - af.data.shift(V, 0, 0))/dx
    Ey = -(af.data.shift(V, 1, 0) - af.data.shift(V, 0, 0))/dy

    return Ex, Ey
