import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import time
from IPython import display

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# Problem Domain
Lx, Ly = (2., 2.)
nx, ny = (200, 200)

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)   # What does 'dealias=3/2' mean????
y_basis = de.Chebyshev('y',ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)	# What is float64???

Reynolds = 1e4
gravity = 1.

problem = de.IVP(domain, variables=['p', 'u', 'v', 'uy', 'vy', 'rho', 'rhoy', 'rhoinit'])
problem.parameters['Re'] = Reynolds
problem.parameters['gr'] = gravity

problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("rhoy - dy(rho) = 0")
# Continuity
problem.add_equation("dx(u) + vy = 0")
# incompressibility equation
problem.add_equation("dt(rho) - (1/Re)*(dx(dx(rho)) + dy(rhoy)) = - u*dx(rho) - v*rhoy")  # diffusion term is necessary to keep 'rho' stable since we are have sharp variation in density. It is 												multiplied with (1/Re) so that it is less significant w.r.t other terms
# x-momentum eqn
problem.add_equation("dt(u) + dx(p) - (1/Re)*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy - (p/rho)*dx(rho)")
# y-momentum eqn
problem.add_equation("dt(v) + dy(p) - (1/Re)*(dx(dx(v)) + dy(vy)) = - u*dx(v) - v*vy - (p/rho)*rhoy - ((rho-rhoinit)*gr)/rho")
# constant rhoinit condition
problem.add_equation("dt(rhoinit) = 0")

# boundary conditions
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(v) = 0.001")				# left and right bc for v are given non-zero to ensure that the pencil matrix doesn't become singular. If left and right of v are 0 det
problem.add_bc("right(v) = -0.001", condition="(nx != 0)")	# of pencil matrix becomes zero because left and right bc for u are zero.
problem.add_bc("left(p) = 0", condition="(nx == 0)")
problem.add_bc("left(dy(rho)) = 0")
problem.add_bc("right(dy(rho)) = 0")

ts = de.timesteppers.RK443

solver =  problem.build_solver(ts)

x = domain.grid(0)	# what does (0) and (1) mean in x and y domain??
y = domain.grid(1)
u = solver.state['u']
uy = solver.state['uy']
v = solver.state['v']
vy = solver.state['vy']
rho = solver.state['rho']
rhoy = solver.state['rhoy']
rhoinit = solver.state['rhoinit']

# Initial conditions
f = h5py.File('analysis_tasks2/analysis_tasks_s1/analysis_tasks_s1_p0.h5','r')	# Path of file from which simulation has to continue. Modify accordingly!
p = f['/tasks/u'][:]
a = len(p)
u['g'] = f['/tasks/u'][(a - 1),:,:]
v['g'] = f['/tasks/v'][(a - 1),:,:]
rhoinit['g'] = 1.0*(np.tanh(y/a)+2.0)
rho['g'] = f['/tasks/rho'][(a - 1),:,:]

u.differentiate('y',out=uy)
v.differentiate('y',out=vy)
rho.differentiate('y',out=rhoy)

solver.stop_sim_time = 3.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.2*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v'))

analysis = solver.evaluator.add_file_handler('analysis_tasks3', sim_dt=0.1, max_writes=50)
analysis.add_task('rho')
analysis.add_task('u')
analysis.add_task('v')
solver.evaluator.vars['Lx'] = Lx
analysis.add_task("integ(rho,'x')/Lx", name='integral rho')

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 10 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()

logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

