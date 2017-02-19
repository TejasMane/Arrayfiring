import pylab as pl
import numpy as np
import h5py


h5f = h5py.File('data_files/time/solution32.h5', 'r')
t32 = h5f['data_files/time/solution_dataset'+str(32)][:]
h5f.close()

h5f = h5py.File('data_files/posa/solution'+str(32)+'.h5', 'r')
pos_ana32 = h5f['data_files/posa/solution_dataset'+str(32)][:]
h5f.close()


h5f = h5py.File('data_files/posn/solution'+str(32)+'.h5', 'r')
pos_num32 = h5f['data_files/posn/solution_dataset'+str(32)][:]
h5f.close()

h5f = h5py.File('data_files/vela/solution'+str(32)+'.h5', 'r')
vel_ana32 = h5f['data_files/vela/solution_dataset'+str(32)][:]
h5f.close()

h5f = h5py.File('data_files/veln/solution'+str(32)+'.h5', 'r')
vel_num32 = h5f['data_files/veln/solution_dataset'+str(32)][:]
h5f.close()





h5f = h5py.File('data_files/time/solution'+str(64)+'.h5', 'r')
t64 = h5f['data_files/time/solution_dataset'+str(64)][:]
h5f.close()
h5f = h5py.File('data_files/posa/solution'+str(64)+'.h5', 'r')
pos_ana64 = h5f['data_files/posa/solution_dataset'+str(64)][:]
h5f.close()
h5f = h5py.File('data_files/posn/solution'+str(64)+'.h5', 'r')
pos_num64 = h5f['data_files/posn/solution_dataset'+str(64)][:]
h5f.close()
h5f = h5py.File('data_files/vela/solution'+str(64)+'.h5', 'r')
vel_ana64 = h5f['data_files/vela/solution_dataset'+str(64)][:]
h5f.close()
h5f = h5py.File('data_files/veln/solution'+str(64)+'.h5', 'r')
vel_num64 = h5f['data_files/veln/solution_dataset'+str(64)][:]
h5f.close()



h5f = h5py.File('data_files/time/solution'+str(128)+'.h5', 'r')
t128 = h5f['data_files/time/solution_dataset'+str(128)][:]
h5f.close()
h5f = h5py.File('data_files/posa/solution'+str(128)+'.h5', 'r')
pos_ana128 = h5f['data_files/posa/solution_dataset'+str(128)][:]
h5f.close()
h5f = h5py.File('data_files/posn/solution'+str(128)+'.h5', 'r')
pos_num128 = h5f['data_files/posn/solution_dataset'+str(128)][:]
h5f.close()
h5f = h5py.File('data_files/vela/solution'+str(128)+'.h5', 'r')
vel_ana128 = h5f['data_files/vela/solution_dataset'+str(128)][:]
h5f.close()
h5f = h5py.File('data_files/veln/solution'+str(128)+'.h5', 'r')
vel_num128 = h5f['data_files/veln/solution_dataset'+str(128)][:]
h5f.close()

#
h5f = h5py.File('data_files/time/solution'+str(256)+'.h5', 'r')
t256 = h5f['data_files/time/solution_dataset'+str(256)][:]
h5f.close()
h5f = h5py.File('data_files/posa/solution'+str(256)+'.h5', 'r')
pos_ana256 = h5f['data_files/posa/solution_dataset'+str(256)][:]
h5f.close()
h5f = h5py.File('data_files/posn/solution'+str(256)+'.h5', 'r')
pos_num256 = h5f['data_files/posn/solution_dataset'+str(256)][:]
h5f.close()
h5f = h5py.File('data_files/vela/solution'+str(256)+'.h5', 'r')
vel_ana256 = h5f['data_files/vela/solution_dataset'+str(256)][:]
h5f.close()
h5f = h5py.File('data_files/veln/solution'+str(256)+'.h5', 'r')
vel_num256 = h5f['data_files/veln/solution_dataset'+str(256)][:]
h5f.close()

#
h5f = h5py.File('data_files/time/solution'+str(512)+'.h5', 'r')
t512 = h5f['data_files/time/solution_dataset'+str(512)][:]
h5f.close()
h5f = h5py.File('data_files/posa/solution'+str(512)+'.h5', 'r')
pos_ana512 = h5f['data_files/posa/solution_dataset'+str(512)][:]
h5f.close()
h5f = h5py.File('data_files/posn/solution'+str(512)+'.h5', 'r')
pos_num512 = h5f['data_files/posn/solution_dataset'+str(512)][:]
h5f.close()
h5f = h5py.File('data_files/vela/solution'+str(512)+'.h5', 'r')
vel_ana512 = h5f['data_files/vela/solution_dataset'+str(512)][:]
h5f.close()
h5f = h5py.File('data_files/veln/solution'+str(512)+'.h5', 'r')
vel_num512 = h5f['data_files/veln/solution_dataset'+str(512)][:]
h5f.close()
#
#
#
# h5f = h5py.File('time/solution'+str(1024)+'.h5', 'r')
# t1024 = h5f['time/solution_dataset'+str(1024)][:]
# h5f.close()
# h5f = h5py.File('posa/solution'+str(1024)+'.h5', 'r')
# pos_ana1024 = h5f['posa/solution_dataset'+str(1024)][:]
# h5f.close()
# h5f = h5py.File('posn/solution'+str(1024)+'.h5', 'r')
# pos_num1024 = h5f['posn/solution_dataset'+str(1024)][:]
# h5f.close()
# h5f = h5py.File('vela/solution'+str(1024)+'.h5', 'r')
# vel_ana1024 = h5f['vela/solution_dataset'+str(1024)][:]
# h5f.close()
# h5f = h5py.File('veln/solution'+str(1024)+'.h5', 'r')
# vel_num1024 = h5f['veln/solution_dataset'+str(1024)][:]
# h5f.close()


pl.plot(t512,pos_num512[:,0], label = '$\mathrm{Numerical}$')
pl.plot(t512,pos_ana512[:,0],'--', label = '$\mathrm{Analytical}$')
pl.legend().draggable()
pl.xlabel('$t$')
pl.ylabel('$\mathrm{Error}$')
pl.ylim(0,1)
pl.show()
pl.clf()

pl.plot(t512,pos_num512[:,1], label = '$\mathrm{Numerical}$')
pl.plot(t512,pos_ana512[:,1],'--', label = '$\mathrm{Analytical}$')
pl.legend().draggable()
pl.xlabel('$t$')
pl.ylabel('$\mathrm{Error}$')
pl.ylim(0,1)
pl.show()
pl.clf()


pl.plot(t512,vel_num512[:,0], label = '$\mathrm{Numerical}$')
pl.plot(t512,vel_ana512[:,0],'--', label = '$\mathrm{Analytical}$')
pl.legend().draggable()
pl.xlabel('$t$')
pl.ylabel('$\mathrm{Error}$')
pl.ylim(0,0.5)
pl.show()
pl.clf()


pl.plot(t512,vel_num512[:,1], label = '$\mathrm{Numerical}')
pl.plot(t512,vel_ana512[:,1],'--', label = '$\mathrm{Analytical}$')
pl.legend().draggable()
pl.xlabel('$t$')
pl.ylabel('$\mathrm{Error}$')
pl.ylim(-0.5,0.5)
pl.show()
pl.clf()





pl.plot(t32,abs(pos_ana32[:,0]-pos_num32[:,0]), label = '$N=32$')
pl.plot(t64,abs(pos_ana64[:,0]-pos_num64[:,0]), label = '$N=64$')
pl.plot(t128,abs(pos_ana128[:,0]-pos_num128[:,0]), label = '$N=128$')
pl.plot(t256,abs(pos_ana256[:,0]-pos_num256[:,0]), label = '$N=256$')
pl.plot(t512,abs(pos_ana512[:,0]-pos_num512[:,0]), label = '$N=512$')
# pl.plot(t1024,abs(pos_ana1024[:,0]-pos_num1024[:,0]), label = '$N=1024$')
# pl.ylim(0,1)
pl.legend().draggable()
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{Error\;in\;x}$')
pl.title('$\mathrm{Error\;in\;x\;vs\;t}$')
pl.show()
pl.clf()



pl.plot(t32,abs(pos_ana32[:,1]-pos_num32[:,1]), label = '$N=32$')
pl.plot(t64,abs(pos_ana64[:,1]-pos_num64[:,1]), label = '$N=64$')
pl.plot(t128,abs(pos_ana128[:,1]-pos_num128[:,1]), label = '$N=128$')
pl.plot(t256,abs(pos_ana256[:,1]-pos_num256[:,1]), label = '$N=256$')
pl.plot(t512,abs(pos_ana512[:,1]-pos_num512[:,1]), label = '$N=512$')
# pl.plot(t1024,abs(pos_ana1024[:,1]-pos_num1024[:,1]), label = '$N=1024$')
# pl.ylim(0,1)
pl.legend().draggable()
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{Error\;in\;y}$')
pl.title('$\mathrm{Error\;in\;y\;vs\;t}$')
pl.show()
pl.clf()



pl.plot(t32,abs(vel_ana32[:,0]-vel_num32[:,0]), label = '$N=32$')
pl.plot(t64,abs(vel_ana64[:,0]-vel_num64[:,0]), label = '$N=64$')
pl.plot(t128,abs(vel_ana128[:,0]-vel_num128[:,0]), label = '$N=128$')
pl.plot(t256,abs(vel_ana256[:,0]-vel_num256[:,0]), label = '$N=256$')
pl.plot(t512,abs(vel_ana512[:,0]-vel_num512[:,0]), label = '$N=512$')
# pl.plot(t1024,abs(vel_ana1024[:,0]-vel_num1024[:,0]), label = '$N=1024$')
pl.ylim(0,0.5)
pl.legend().draggable()
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{Error\;in\;v_x}$')
pl.title('$\mathrm{Error\;in\;v_x\;vs\;t}$')
pl.show()
pl.clf()



pl.plot(t32,abs(vel_ana32[:,1]-vel_num32[:,1]), label = '$N=32$')
pl.plot(t64,abs(vel_ana64[:,1]-vel_num64[:,1]), label = '$N=64$')
pl.plot(t128,abs(vel_ana128[:,1]-vel_num128[:,1]), label = '$N=128$')
pl.plot(t256,abs(vel_ana256[:,1]-vel_num256[:,1]), label = '$N=256$')
pl.plot(t512,abs(vel_ana512[:,1]-vel_num512[:,1]), label = '$N=512$')
# pl.plot(t1024,abs(vel_ana1024[:,1]-vel_num1024[:,1]), label = '$N=1024$')
pl.ylim(0,0.5)
pl.legend().draggable()
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{Error\;in\;v_y}$')
pl.title('$\mathrm{Error\;in\;v_y\;vs\;t}$')
pl.show()
pl.clf()


N = np.array( [32, 64 , 128, 256, 512] )



errorx = np.zeros(len(N), dtype = np.float)

errorx[0] = (abs(pos_ana32[-10,0]-pos_num32[-10,0]))/len(t32)
errorx[1] = (abs(pos_ana64[-10,0]-pos_num64[-10,0]))/len(t64)
errorx[2] = (abs(pos_ana128[-10,0]-pos_num128[-10,0]))/len(t128)
errorx[3] = (abs(pos_ana256[-10,0]-pos_num256[-10,0]))/len(t256)
errorx[4] = (abs(pos_ana512[-10,0]-pos_num512[-10,0]))/len(t512)
# errorx[5] = (abs(pos_ana1024[-10,0]-pos_num1024[-10,0]))/len(t1024)


pl.loglog(N,errorx,'-o',label='$\mathrm{Numerical}$')
pl.loglog(N,1.5*(N**-1.999),'--',color = 'black',lw = 2,label = ' $O(N^{-2})$ ')
pl.legend().draggable()
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{Error\;in\;x}$')
pl.title('$\mathrm{Error\;in\;x\;vs\;N}$')
pl.show()
pl.clf()



errory = np.zeros(len(N), dtype = np.float)

errory[0] = (abs(pos_ana32[-10,1]-pos_num32[-10,1]))/len(t32)
errory[1] = (abs(pos_ana64[-10,1]-pos_num64[-10,1]))/len(t64)
errory[2] = (abs(pos_ana128[-10,1]-pos_num128[-10,1]))/len(t128)
errory[3] = (abs(pos_ana256[-10,1]-pos_num256[-10,1]))/len(t256)
errory[4] = (abs(pos_ana512[-10,1]-pos_num512[-10,1]))/len(t512)
# errory[5] = (abs(pos_ana1024[-10,1]-pos_num1024[-10,1]))/len(t1024)


pl.loglog(N,errory,'-o',label='$\mathrm{Numerical}$')
pl.loglog(N,1.5*(N**-1.999),'--',color = 'black',lw = 2,label = ' $O(N^{-2})$ ')
pl.legend().draggable()
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{Error\;in\;y}$')
pl.title('$\mathrm{Error\;in\;y\;vs\;N}$')
pl.show()
pl.clf()

errorvx = np.zeros(len(N), dtype = np.float)

errorvx[0] = (abs(vel_ana32[-10,0]-vel_num32[-10,0]))/len(t32)
errorvx[1] = (abs(vel_ana64[-10,0]-vel_num64[-10,0]))/len(t64)
errorvx[2] = (abs(vel_ana128[-10,0]-vel_num128[-10,0]))/len(t128)
errorvx[3] = (abs(vel_ana256[-10,0]-vel_num256[-10,0]))/len(t256)
errorvx[4] = (abs(vel_ana512[-10,0]-vel_num512[-10,0]))/len(t512)
# errorvx[5] = (abs(vel_ana1024[-10,0]-vel_num1024[-10,0]))/len(t1024)



pl.loglog(N,errorvx,'-o',label='$\mathrm{Numerical}$')
pl.loglog(N,1.5*(N**-1.999),'--',color = 'black',lw = 2,label = ' $O(N^{-2})$ ')
pl.legend().draggable()
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{Error\;in\;v_x}$')
pl.title('$\mathrm{Error\;in\;v_x\;vs\;N}$')
pl.show()
pl.clf()


errorvy = np.zeros(len(N), dtype = np.float)

errorvy[0] = (abs(vel_ana32[-10,1]-vel_num32[-10,1]))/len(t32)
errorvy[1] = (abs(vel_ana64[-10,1]-vel_num64[-10,1]))/len(t64)
errorvy[2] = (abs(vel_ana128[-10,1]-vel_num128[-10,1]))/len(t128)
errorvy[3] = (abs(vel_ana256[-10,1]-vel_num256[-10,1]))/len(t256)
errorvy[4] = (abs(vel_ana512[-10,1]-vel_num512[-10,1]))/len(t512)
# errorvy[5] = (abs(vel_ana1024[-10,1]-vel_num1024[-10,1]))/len(t1024)



pl.loglog(N,errorvy,'-o',label='$\mathrm{Numerical}$')
pl.loglog(N,1.5*(N**-1.999),'--',color = 'black',lw = 2,label = ' $O(N^{-2})$ ')
pl.legend().draggable()
pl.xlabel('$\mathrm{N}$')
pl.ylabel('$\mathrm{Error\;in\;v_y}$')
pl.title('$\mathrm{Error\;in\;v_y\;vs\;N}$')
pl.show()
pl.clf()

