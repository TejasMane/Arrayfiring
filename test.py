import arrayfire as af
import numpy as np
from scipy import signal


#dx = 1

##f = np.matrix('-0.5, 0, 0.5, 1, 1.5; -0.5, 0, 0.5, 1, 1.5; -0.5, 0, 0.5, 1, 1.5; -0.5, 0, 0.5, 1, 1.5; -0.5, 0, 0.5, 1, 1.5')


##f= np.array([-0.5, 0, 0.5, 1, 1.5])



#a = af.randu(3, 3)

##cf = np.convolve(f, [0, 1, 0], 'same') / dx


##cf = signal.convolve(f, [0, 1, 0], 'same') / dx


#x1 = af.data.constant(0, 3)
#x1[1] = 1
#x2 = af.data.constant(0, 3)
#x2[0] = 1
#x2[1] = -1


#cf = af.signal.convolve2_separable(x1, x2, a)


#print('a is ',a)
#print('a convolve is ',cf)

##af.signal.convolve()



data = np.zeros((3, 3), dtype = np.float)
data[1,:] = 2
data[2,:] = 3

posx = np.ones(2,dtype = np.float)



posy = np.ones(2,dtype = np.float)
posy[1] = 1.9

forward_difference = np.array([0,1,-1])
backward_difference = np.array([0,1,-1])

Forward_x, Forward_y = np.meshgrid(forward_difference, forward_difference)
backward_x, backward_y = np.meshgrid(backward_difference, backward_difference)


print('Forward_y = ', Forward_y)
results = signal.convolve2d(data, [[0], [1], [-1]], 'same')




print('Data = ', data)

print('results = ', results)
 
