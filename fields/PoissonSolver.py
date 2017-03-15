import arrayfire as af
import numpy as np
import h5py
# from wall_options.EM_periodic import periodic

def Dirichlet(V, ghost_cells, x_points, y_points):

    V[ 0 : ghost_cells, :]                    = V[y_points -1 - 2 * ghost_cells: y_points -1 - 1 * ghost_cells, :]
    V[ :, 0 : ghost_cells]                    = V[:, x_points -1 - 2 * ghost_cells: x_points -1 - 1 * ghost_cells]
    V[y_points - ghost_cells : y_points, :]   = V[ghost_cells + 1: 2 * ghost_cells + 1, :]
    V[:, x_points - ghost_cells : x_points]   = V[: , ghost_cells + 1: 2 * ghost_cells + 1]

    return V

# Successive over relaxation Poisson solver

def SOR(rho_with_ghost, ghost_cells, dx, dy, *args, **kwargs):
    # print('dx', dx)
    # input('check')
    x_points = (rho_with_ghost[0, :]).elements()
    y_points = (rho_with_ghost[:, 0]).elements()

    omega   = kwargs.get('omega', None)
    epsilon = kwargs.get('epsilon', None)
    max_iterations = kwargs.get('max_iterations', None)

    l = x_points

    if(omega == None):
        omega = 2/(1+(np.pi/l)) - 1

    if(epsilon == None):
        epsilon = 1e-12

    if(max_iterations == None):
        max_iterations = 50000

    X_physical_index = ghost_cells + af.data.range(y_points - 2 * ghost_cells, d1= x_points - 2 * ghost_cells, dim=1)
    Y_physical_index = ghost_cells + af.data.range(y_points - 2 * ghost_cells, d1= x_points - 2 * ghost_cells, dim=0)
    x1 = af.data.range(y_points, dim=0)

    V_k      = af.data.constant(0, (rho_with_ghost[:, 0]).elements(), (rho_with_ghost[0, :]).elements(), dtype=af.Dtype.f64)
    Error      = af.data.constant(0,int(max_iterations/100) , dtype=af.Dtype.f64)
    V_k = Dirichlet(V_k, ghost_cells, x_points, y_points)

    V_k_plus = af.data.constant(0, (rho_with_ghost[:, 0]).elements(), (rho_with_ghost[0, :]).elements(), dtype=af.Dtype.f64)

    V_k_plus = Dirichlet(V_k_plus, ghost_cells, x_points, y_points)
    # omega = 0.6

    V_k_plus =  (1-omega) * V_k \
                                + (omega/(2*dx**2 + 2*dy**2)) \
                                * (    (dy**2)*(af.data.shift(V_k,0,1) + af.data.shift(V_k,0,-1)) \
                                    + (dx**2)*(af.data.shift(V_k,1,0) + af.data.shift(V_k,-1,0)) \
                                    + ((dx**2)*dy**2)*(rho_with_ghost) \
                                  )

    V_k_plus = Dirichlet(V_k_plus, ghost_cells, x_points, y_points)

    for i in range(max_iterations):

        V_k = V_k_plus.copy()

        V_k_plus =  (1-omega) * V_k \
                                    + (omega/(2*dx**2 + 2*dy**2)) \
                                    * (    (dy**2)*(af.data.shift(V_k,0,1) + af.data.shift(V_k,0,-1)) \
                                        + (dx**2)*(af.data.shift(V_k,1,0) + af.data.shift(V_k,-1,0)) \
                                        + ((dx**2)*dy**2)*(rho_with_ghost) \
                                      )

        V_k_plus = Dirichlet(V_k_plus, ghost_cells, x_points, y_points)


        if(i%10 == 0):
            if(i%100==0):
                print('iteration = ',i, 'Poisson convergence = ', af.max(af.abs(V_k_plus - V_k)) )
                Error[i/100] = af.sum(af.abs(V_k_plus - V_k))/(x_points*y_points)
                # print('Error',Error[:10])
            if(af.max(af.abs(V_k_plus - V_k)) < epsilon):
                h5f = h5py.File('data_files/error.h5', 'w')
                h5f.create_dataset('Error',   data = Error)
                h5f.close()
                print('iteration = ',i, 'Poisson convergence = ', af.max(af.abs(V_k_plus - V_k)) )
                print('epsilon is ', epsilon)
                return V_k_plus


    h5f = h5py.File('data_files/error.h5', 'w')
    h5f.create_dataset('Error',   data = Error)
    h5f.close()
    af.eval(V_k_plus)
    print('iteration = ',i, 'Poisson convergence = ', af.max(af.abs(V_k_plus - V_k)) )
    return V_k_plus


def FFT_1D(rho_with_ghost, ghost_cells, dx, dy):

    x_points = (rho_with_ghost[0, :]).elements()
    y_points = (rho_with_ghost[:, 0]).elements()










def compute_Electric_field(V, dx, dy, ghost_cells):
    # Ex[i, j] = -\nabla\;V = -(V[i,j + 1] - V[i,j])/dx
    # Ey[i,j] = -\nabla\;V = - (V[i + 1, j] - V[i, j])/dy
    # Row column representation
    x_points = (V[0, :]).elements()
    y_points = (V[:, 0]).elements()

    Ex = -(af.data.shift(V, 0, 1) - af.data.shift(V, 0, 0))/dx
    Ey = -(af.data.shift(V, 1, 0) - af.data.shift(V, 0, 0))/dy

    Ex = Dirichlet(Ex, ghost_cells, x_points, y_points)
    Ey = Dirichlet(Ey, ghost_cells, x_points, y_points)

    return Ex, Ey

def compute_divergence_E_minus_rho(Ex, Ey, rho, dx, dy, ghost_cells):
    # (Ex(i + 1/2, j)- Ex(i - 1/2, j))/dx + (Ey(i, j + 1/2)- Ey(i, j - 1/2))/dy - rho[i, j]
    # Row column representation
    # (Ex[i, j] - Ex[i, j-1])/dx + (Ey[i, j] - Ey[i - 1, j])/dy - rho[i, j]
    x_points = (rho[0, :]).elements()
    y_points = (rho[:, 0]).elements()


    div_E_minus_rho =   (af.data.shift(Ex, 0, 0) - af.data.shift(Ex, 0, -1))/dx \
                      + (af.data.shift(Ey, 0, 0) - af.data.shift(Ey, -1, 0))/dy\
                      - (rho)

    div_E_minus_rho = Dirichlet(div_E_minus_rho, ghost_cells, x_points, y_points)


    return div_E_minus_rho
