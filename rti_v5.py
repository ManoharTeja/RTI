'''
Inviscid
'''

import numpy as np
import h5py
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# Problem Domain
Lx, Ly = (1., 1.)
nx, ny = (100, 100)

# ************ Create bases and domain ************************
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Chebyshev('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
# *************************************************************

problem = de.IVP(domain, variables=['p', 'u', 'v', 'uy', 'vy', 'rho', 'rhoy'])

# ********** Parameters *********
problem.parameters['Re'] = 1e3
problem.parameters['gr'] = 1.
# Inserting a Non-constant Cofficient (NCC) as parameter
y = domain.grid(1)
rhoinit = domain.new_field()
rhoinit.meta['x']['constant'] = True
rhoinit['g'] = 1.0*(np.tanh(y/0.05)+2.0)
problem.parameters['rhoinit'] = rhoinit
# *********************************

# ***************** PDEs to solve ****************************
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("rhoy - dy(rho) = 0")
# Imcompressibility equation
problem.add_equation("dx(u) + vy = 0")
# Continuity equation
problem.add_equation("dt(rho) + v*dy(rhoinit) = - u*dx(rho) - v*(rhoy)")
# x-momentum eqn
problem.add_equation("dt(u) + dx(p) = - u*dx(u) - v*uy - (p/(rho+rhoinit))*dx(rho)")
# y-momentum eqn
problem.add_equation("dt(v) + dy(p) = - u*dx(v) - v*vy - (p/(rho+rhoinit))*(rhoy+dy(rhoinit)) - (rho*gr)/(rho+rhoinit)")
# ************************************************************

# ******** Boundary conditions **************
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("left(p) = 0", condition="(nx == 0)")
problem.add_bc("left(rho) = 0")
#problem.add_bc("right(rho) = 0")
# *******************************************

ts = de.timesteppers.RK443

solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
u = solver.state['u']
uy = solver.state['uy']
v = solver.state['v']
vy = solver.state['vy']
rho = solver.state['rho']
rhoy = solver.state['rhoy']

# ******************** Initial conditions ************************
a = 0.05
sigma = 0.2
amp = -0.02
u['g'] = -((amp*(Lx/2)*y)/(np.pi*sigma*sigma))*np.cos((2.0*np.pi*x)/(Lx/2))*np.exp(-(y*y)/(sigma*sigma))	# Lx = 2
v['g'] = amp*np.sin((2.0*np.pi*x)/(Lx/2))*np.exp(-(y*y)/(sigma*sigma))					# Lx = 2
#u['g'] = -((amp*Lx*y)/(np.pi*sigma*sigma))*np.cos((2.0*np.pi*x)/Lx)*np.exp(-(y*y)/(sigma*sigma))		# Lx = 1
#v['g'] = amp*np.sin((2.0*np.pi*x)/Lx)*np.exp(-(y*y)/(sigma*sigma))						# Lx = 1
rho['g'] = 1.0*(np.tanh(y/a)+2.0)
# *****************************************************************


u.differentiate('y',out=uy)
v.differentiate('y',out=vy)
rho.differentiate('y',out=rhoy)

solver.stop_sim_time = 6.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.002*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v'))

analysis = solver.evaluator.add_file_handler('analysis_tasks5', sim_dt=0.1, max_writes=50)
analysis.add_task('rho')
analysis.add_task('u')
analysis.add_task('v')
solver.evaluator.vars['Lx'] = Lx
analysis.add_task("integ(rho,'x')/Lx", name='integral rho_x')

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

