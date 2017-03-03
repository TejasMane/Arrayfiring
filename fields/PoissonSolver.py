import arrayfire as af
import numpy as np

def generate_Poisson_matrix(no_of_elements_x, no_of_elements_y):

    Ax = af.data.constant(0, no_of_elements_y, no_of_elements_x, dtype=af.Dtype.f64)
    Ay = af.data.constant(0, no_of_elements_y, no_of_elements_x, dtype=af.Dtype.f64)

    Index_x = np.arange(1, no_of_elements_x - 1, 1)
    Index_y = np.arange(1, no_of_elements_y - 1, 1)

    second_order_vector = af.Array([1, -2, 1])
    second_order_vector_x = af.tile(af.reorder(second_order_vector, 1), no_of_elements_y, 1)
    second_order_vector_y = af.tile(second_order_vector, 1, no_of_elements_x)

    Ax[0:1, 0] = af.Array([-2.0, 1])
    Ax[-1, 0] = af.Array([1.0])
    Ax[0:1, 0] = af.Array([-2.0, 1])
    Ax[-1, 0] = af.Array([1.0])

    Ay[0:1, 0] = af.Array([-2.0, 1])
    Ay[-1, 0] = af.Array([1.0])
    Ay[0:1, 0] = af.Array([-2.0, 1])
    Ay[-1, 0] = af.Array([1.0])

    Ax[Index_y, Index_x : Index_x + 3] = second_order_vector_x
    Ay[Index_x, Index_y : Index_y + 3] = second_order_vector_y

    return Ax, Ay



def SOR(Ax, Ay, rho, epsilon):

    V_k      = af.data.constant(0, (rho[:, 0]).elements(), (rho[0, :]).elements(), dtype=af.Dtype.f64)
    V_k_plus = af.data.constant(0, (rho[:, 0]).elements(), (rho[0, :]).elements(), dtype=af.Dtype.f64)

    l = (V[:, 0]).elements()

    omega = 2/(1+(np.pi/l))

    Lx = 0
    Ux = 0
    Dx = 0

    Ly = 0
    Uy = 0
    Dy = 0

    L_omega_x = -1 * af.inverse(Dx + omega * Lx) * (omega * Ux + (omega - 1) * Dx)
    c_x = af.inverse(Dx + omega * Lx) * omega * rho

    L_omega_y = -1 * af.inverse(Dy + omega * Ly) * (omega * Uy + (omega - 1) * Dy)
    c_y = af.inverse(Dy + omega * Ly) * omega * rho

    while(af.sum(af.abs(V_k_plus - V_k))<epsilon):

        V_k = V_k_plus

        V_k_plus =  (L_omega_x * V_k + c_x) + (L_omega_y * V_k + c_y)

    return V_k_plus
