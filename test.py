import time
import numpy as np
import arrayfire as af
#af.set_backend("cpu")

# This is a benchmark test to compare numpy and arrayfire:

print("The following line displays the ArrayFire build and device details:")
af.info()


aNumPy = np.random.rand(50, 50)
bNumPy = np.random.rand(50, 50)

np_time_start = time.time()

for i in range(10):
  cNumPy = aNumPy + bNumPy

np_time_end     = time.time()
np_time_elapsed = np_time_end - np_time_start

print("numpy implementation run took time =", np_time_elapsed," seconds")

aArrayFire = af.Array(aNumPy.ctypes.data, aNumPy.shape, aNumPy.dtype.char)
bArrayFire = af.Array(bNumPy.ctypes.data, bNumPy.shape, bNumPy.dtype.char)

kernel_compilation_time_start = time.time()

cArrayFire = aArrayFire + bArrayFire
af.eval(cArrayFire)
af.sync()

kernel_compilation_time_end     = time.time()
kernel_compilation_time_elapsed = kernel_compilation_time_end - kernel_compilation_time_start

print("Kernel compilation complete. Compilation time = ", kernel_compilation_time_elapsed)

af_time_start = time.time()

for i in range(1000000):
  cArrayFire = aArrayFire + bArrayFire
  af.eval(cArrayFire)
 
af.sync()
af_time_end     = time.time()
af_time_elapsed = af_time_end - af_time_start

print("arrayfire implementation run took time =", af_time_elapsed," seconds")
 
