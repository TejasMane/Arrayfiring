import numpy as np
import h5py
import params
import pylab as pl
import arrayfire as af
import initialize
from scipy.integrate import odeint


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
a = (a / (no_of_particles / x_divisions))
pl.plot(x_temp,a)
pl.xlabel('$x$')
pl.ylabel(r'$\delta\rho(x)$')
pl.ylim(0.0,2.0)
# pl.savefig('data_files/images/' + '%04d'%(0) + '.png')
pl.show()
pl.clf()


# print(max(vel_x_initial))
# print(min(vel_x_initial))
# x_temp = np.linspace(0,1,100)
# a, b = np.histogram(np.sqrt(vel_x_initial**2+vel_y_initial**2+vel_z_initial**2), bins=(x_divisions))
# # a = (a / (no_of_particles / x_divisions))-1
# pl.plot(x_temp,a)
# pl.xlabel('$x$')
# pl.ylabel('Velocity')
# pl.show()
# pl.clf()


data = np.zeros((time.size), dtype = np.float)

divisions_for_histogram_x = x_divisions/10

divisions_for_histogram = divisions_for_histogram_x




# x_temp = np.linspace(0,1,100)

for time_index,t0 in enumerate(time):

    if(time_index%100 ==0):
        print(time_index)
    if(time_index == time.size -1):
        break
    # if (time_index % 10 == 0):
    h5f = h5py.File('data_files/timestepped_data/solution_'+str(time_index)+'.h5', 'r')
    x_coords = h5f['x_coords'][:]
    y_coords = h5f['y_coords'][:]
    h5f.close()


    a,b = np.histogram(x_coords, bins=(divisions_for_histogram), range = (0,1))
    data[time_index] = max(abs(a/(no_of_particles/divisions_for_histogram_x)))-1
    # print('time = ', time_index, 'Amplitude = ', max(a))
    # pl.plot(a)
    # pl.savefig('data_files/images/' + '%04d' % (time_index) + '.png')
    # pl.clf


# Setting the variables in the maxwell distribution
m = 1
K = 1
T = 1
e =-1
# In[ ]:

# k for the mode in fourier space
k = 2*np.pi


# In[ ]:

# The maxwell Boltzman function
def f_0(v):
    return np.sqrt(m/(2*np.pi*K*T))*np.exp(-m*v**2/(2*K*T))

# This the function which returns the derivative of the maxwell boltzmann equation
def diff_f_0_v(v):
    return np.sqrt(m/(2*np.pi*K*T))*np.exp(-m*v**2/(2*K*T)) * ( -m * v / (K * T))


# In[ ]:

# Assign the maxim and minimum velocity for the velocity grid
velocity_max =  +10
velocity_min =  -10

# Set the divisions for the velocity grid
number_of_velocities_points = 1001
velocity_x = np.linspace(velocity_min, velocity_max, number_of_velocities_points)
dv = velocity_x[1] - velocity_x[0]


# In[ ]:

# Function that returns df_i/dt and df_r/dt used for odeint function
# See the latex document for more details on the differential equations
# This has been done to split the imaginary and real part of the ODE
def diff_delta_f(Y,t):
    f_r = Y[0:len(velocity_x)]  # Initial conditions for odeint
    f_i = Y[len(velocity_x): 2 * len(velocity_x)]

    int_Df_i = np.sum(f_i) * (velocity_x[1]-velocity_x[0])
    int_Df_r = np.sum(f_r) * (velocity_x[1]-velocity_x[0])

    # This the derivate for f_r and f_i given in the latex document
    dYdt =np.concatenate([(k * velocity_x * f_i) - e*(int_Df_i * diff_f_0_v(velocity_x)/k ), \
                           -(k * velocity_x * f_r) + e*(int_Df_r * diff_f_0_v(velocity_x)/k )\
                         ], axis = 0)
    # This returns the derivative for the coupled set of ODE

    return dYdt

def diff_delta_f_Ex(Y,t):

    f_r = Y[0:len(velocity_x)]  # Initial conditions for odeint
    f_i = Y[len(velocity_x): 2 * len(velocity_x)]
    E_x_r = Y[2 * len(velocity_x)]
    E_x_i = Y[2 * len(velocity_x) + 1]

    int_v_delta_f_dv_i = np.sum(f_i * velocity_x) * (dv)
    int_v_delta_f_dv_r = np.sum(f_r * velocity_x) * (dv)
    int_v_delta_f_dv = np.array([int_v_delta_f_dv_r, int_v_delta_f_dv_i ] )

    # This the derivate for f_r and f_i given in the latex document
    dYdt =np.concatenate([(    k * velocity_x * f_i) - e*(E_x_r * diff_f_0_v(velocity_x) ), \
                            - (k * velocity_x * f_r) - e*(E_x_i * diff_f_0_v(velocity_x) ), \
                                -1 * int_v_delta_f_dv\
                         ], axis = 0\
                        )
    # This returns the derivative for the coupled set of ODE

    return dYdt

# In[ ]:

# Set the initial conditions for delta f(v,t) here
delta_f_initial = np.zeros((2 * len(velocity_x)), dtype = np.float)
delta_f_initial[0: len(velocity_x)] = 0.5 * f_0(velocity_x)

delta_f_Ex_initial = np.zeros((2 * len(velocity_x)+2), dtype = np.float)
delta_f_Ex_initial[0 : len(velocity_x)] = 0.5 * f_0(velocity_x)
delta_f_Ex_initial[2 * len(velocity_x) + 1] = -1 * (1/k) * np.sum(delta_f_Ex_initial[0: len(velocity_x)] ) * dv

# In[ ]:

# Setting the parameters for time here
final_time = 40
dt = 0.001
time_ana = np.arange(0, final_time, dt)


# In[ ]:

# Variable for temperorily storing the real and imaginary parts of delta f used for odeint
initial_conditions_delta_f = np.zeros((2 * len(velocity_x)), dtype = np.float)
old_delta_f = np.zeros((2 * len(velocity_x)), dtype = np.float)


initial_conditions_delta_f_Ex = np.zeros((2 * len(velocity_x) + 2), dtype = np.float)
old_delta_f_Ex = np.zeros((2 * len(velocity_x) + 2 ), dtype = np.float)
# Variable for storing delta rho

delta_rho1 = np.zeros(len(time_ana), dtype = np.float)
delta_rho2 = np.zeros(len(time_ana), dtype = np.float)
delta_f_temp = np.zeros(2 * len(velocity_x), dtype=np.float)
temperory_delta_f_Ex = np.zeros(2 * len(velocity_x) + 2, dtype=np.float)
# In[ ]:

for time_index, t0 in enumerate(time_ana):
    if(time_index%1000==0):
        print("Computing for TimeIndex = ", time_index)
    t0 = time_ana[time_index]
    if (time_index == time_ana.size - 1):
        break
    t1 = time_ana[time_index + 1]
    t = [t0, t1]

    # delta f is defined on the velocity grid


    # Initial conditions for the odeint
    if(time_index == 0):
        # Initial conditions for the odeint for the 2 ODE's respectively for the first time step
        # First column for storing the real values of delta f and 2nd column for the imaginary values
        initial_conditions_delta_f                 = delta_f_initial.copy()
        initial_conditions_delta_f_Ex                 = delta_f_Ex_initial.copy()
        # Storing the integral sum of delta f dv used in odeint

    else:
        # Initial conditions for the odeint for the 2 ODE's respectively for all other time steps
        # First column for storing the real values of delta f and 2nd column for the imaginary values
        initial_conditions_delta_f= old_delta_f.copy()
        initial_conditions_delta_f_Ex= old_delta_f_Ex.copy()
        # Storing the integral sum of delta f dv used in odeint

    # Integrating delta f

    temperory_delta_f = odeint(diff_delta_f, initial_conditions_delta_f, t)[1]
    temperory_delta_f_Ex = odeint(diff_delta_f_Ex, initial_conditions_delta_f_Ex, t)[1]

    # Saving delta rho for current time_index
    delta_rho1[time_index] = ((sum(dv * temperory_delta_f[0: len(velocity_x)])))
    delta_rho2[time_index] = ((sum(dv * temperory_delta_f_Ex[0: len(velocity_x)])))

    # Saving the solution for to use it for the next time step
    old_delta_f = temperory_delta_f.copy()
    old_delta_f_Ex = temperory_delta_f_Ex.copy()


h5f = h5py.File('data_files/LT.h5', 'w')
h5f.create_dataset('delta_rho1',   data = delta_rho1)
h5f.create_dataset('delta_rho2',   data = delta_rho2)
h5f.close()


h5f           = h5py.File('data_files/LT.h5', 'r')
delta_rho1     = h5f['delta_rho1'][:]
delta_rho2     = h5f['delta_rho2'][:]
h5f.close()

# print('data is ', data)

# Plotting the required quantities here

pl.plot(time_ana, (abs(delta_rho1)),label = '$\mathrm{Linear\;Theory}$')
pl.plot(time_ana, (abs(delta_rho2)),label = '$\mathrm{Linear\;Theory2}$')
# pl.plot(time,data,label = '$\mathrm{Numerical\;PIC}$')
pl.xlabel('$\mathrm{time}$')
pl.ylabel(r'$\delta \hat{\rho}\left(t\right)$')
pl.title('$\mathrm{Linear\;Landau\;damping}$')
pl.legend()
# pl.ylim(0, 0.7)
# pl.xlim(0,2)
pl.show()

pl.plot(time_ana, np.log(abs(delta_rho1)),label = '$\mathrm{Linear\;Theory}$')
pl.plot(time_ana, np.log(abs(delta_rho2)),label = '$\mathrm{Linear\;Theory}$')
# pl.plot(time,np.log(abs(data)),label = '$\mathrm{Numerical\;PIC}$')
pl.xlabel('$\mathrm{time}$')
# pl.xlim(0,2)
pl.ylabel(r'$\delta \hat{\rho}\left(t\right)$')
pl.title('$\mathrm{Linear\;Landau\;damping}$')
pl.legend()
pl.show()
