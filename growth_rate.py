import h5py
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

Lx, Ly = (2, 2)
nx, ny = (200, 200)

# Read in the data
f = h5py.File('analysis_tasks2/analysis_tasks2_s1/analysis_tasks2_s1_p0.h5','r')

y = f['/scales/y/1.0'][:] # y = f['/scales/y'] gives the sub-groups in group y. 
x = f['/scales/x/1.0'][:]
v = f['/tasks/v'][:]
t = f['scales']['sim_time'][:]
f.close()

h = Lx/nx
vv = 0
vel = [0]*len(v)

for j in range(len(v)):	
	for i in range(nx):
		if i == 0 or i == (nx-1):
			vv = vv + v[j,i,100]*v[j,i,100]
		elif i % 2 == 0:
			vv = vv + 4*v[j,i,100]*v[j,i,100]
		else:
			vv = vv + 2*v[j,i,100]*v[j,i,100]

	vel[j] = (h/3)*vv

# Comparison with analytical method
t1 = np.linspace(0, 5, 50)
y1 = [0]*50
y2 = [0]*50
for i in range(50):
	y1[i] = (0.0005*np.exp(2*np.sqrt(np.pi)*t1[i]/np.sqrt(1.6)))
	y2[i] = 0.0002*np.exp(2*np.sqrt(np.pi)*t1[i])	

plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(12, 6), dpi=80)
plt.semilogy(t, vel, linestyle='-', marker='o', color='r')
plt.plot(t1, y1, linestyle='-', marker='x', color='b')
plt.plot(t1, y2, linestyle='-', marker='x', color='k')
plt.legend(["Simulation", "Analytical with factor", "Analytical without factor"])
plt.grid(True, which="both")
plt.xlabel(r'$t$', fontsize=24)
plt.ylabel(r'$v^2$', fontsize=24)

plt.show()
