import h5py
import numpy as np
import matplotlib.pyplot as plt

Lx, Ly = (1., 1.)
# Read in the data
f = h5py.File('analysis_tasks3/analysis_tasks3_s1/analysis_tasks3_s1_p0.h5','r')
#print(list(f.keys())) # lists the keys available in the file
print(list(f['/scales'])) # lists all sub-groups available in group scales
print(list(f['/tasks'])) # lists all sub-groups available in group tasks

y = f['/scales/y/1.0'][:] # y = f['/scales/y'] gives the sub-groups in group y. 
			   # So, to print the data in y, use the syntax in line 11.
x = f['/scales/x/1.0'][:]
rho = f['/tasks/(rho+rhoinit)'][:]
rhofluc = f['/tasks/rho'][:]
v = f['/tasks/v'][:]
t = f['scales']['sim_time'][:]
f.close()
print(t)
print(t[33])
# Density contour (Contour of scalar field)
xm, ym = np.meshgrid(x,y)
plt.rcParams.update({'font.size': 18})

fig, axis = plt.subplots(figsize=(12,6), dpi=80)
#p = axis.pcolormesh(xm, ym, rho[(len(rho)-1),:,:].T, cmap='RdBu_r') # shows the contour of rho data collected at last time instant len(rho) gives number of time instants at which data is collects  									 	but since data starts with 0 we need to give (len(rho)-1) to represent last time instant
p = axis.pcolormesh(xm, ym, rhofluc[33,:,:].T, cmap='RdBu_r')
axis.set_xlim([0,Lx])
axis.set_ylim([-Ly/2, Ly/2])
plt.xlabel(r'$x$', fontsize=24)
plt.ylabel(r'$y$', fontsize=24)

'''
fig, axis = plt.subplots(figsize=(12,6), dpi=80)
# Form1
#plt.contourf(xm, ym, rho[(len(rho)-1),:,:].T, levels=9, cmap='coolwarm')
plt.contourf(xm, ym, rho[14,:,:].T, levels=9, cmap='coolwarm')
#plt.colorbar()

# Form2
#CS = plt.contourf(xm, ym, rho[(len(rho)-1),:,:].T, levels=7, cmap=cm.coolwarm)
#colorbar = plt.colorbar(CS)

# Form3
#levels = np.linspace(1.0, 3.0, 9)
#CS = plt.contourf(xm, ym, rho[(len(rho)-1),:,:].T, levels=levels, cmap=cm.coolwarm, extend='min')
#colorbar = plt.colorbar(CS)

axis.set_xlim([0,Lx])
axis.set_ylim([-Ly/2, Ly/2])
plt.xlabel(r'$x$', fontsize=24)
plt.ylabel(r'$y$', fontsize=24)

#print(rho.shape) # shows that rho matrix is (40, 96,192), where 40 is represents that 'rho' data is collected at 40 time instants, 96 and 192 represents the 'rho' value at 96, 192 grid points along x and y

rho_x = rho.mean(axis=1) # x-averaging

#print(rho_x.shape)
a = rho_x[28,:]
b = rho_x[49,:]
#print(rho_x.shape) # Now print(rho_x.shape) shows that the rho matrix is (40,192) so x-averaged 'rho' data at 40 time instants the axis 0 now to get 1D data

for i in range(192):
	if a[i] > 2.85:
		y1 = y[i]
		break
		
for i in range(192):
	if a[i] > 1.05:
		y2 = y[i]
		break
	
print(y1)
print(y2)

y_mix1 = y1 - y2

for i in range(192):
	if b[i] > 2.85:
		y3 = y[i]
		break
		
for i in range(192):
	if b[i] > 1.05:
		y4 = y[i]
		break
		
y_mix2 = y3 - y4

dt = t[49] - t[28]

growth_rate = abs(y_mix1-y_mix2)/(dt)
print(growth_rate)

plt.figure(figsize=(12, 6), dpi=80)
for i in range(len(rho_x)):
   plt.plot(y, rho_x[i, :], '-')
plt.xlabel(r'$y$', fontsize=24)
plt.ylabel(r'$\frac{\int \rho dx}{L_x}$',fontsize=24)

plt.figure(figsize=(12, 6), dpi=80)
plt.plot(y, rho_x[(len(rho)-1), :], '-')
plt.xlabel(r'$y$', fontsize=24)
plt.ylabel(r'$\frac{\int \rho dx}{L_x}$',fontsize=24)
'''
plt.show()

