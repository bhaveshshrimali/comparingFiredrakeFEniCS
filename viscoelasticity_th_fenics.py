# Preliminaries

import numpy as np
import matplotlib.pyplot as plt
from dolfin import *

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5
parameters["form_compiler"]["representation"] = "uflacs"
parameters["linear_algebra_backend"] = "PETSc"
ffc_options = {
    "optimize": True,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True}
set_log_level(30)
comm = MPI.comm_world
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",  # lu or gmres or cg or bicgstab 'preconditioner: ilu, amg, jacobi'
                                          "preconditioner": "petsc_amg",
                                          "maximum_iterations": 50,
                                          "report": True,
                                          "line_search": 'basic',
                                          "error_on_nonconvergence": False,
                                          "relative_tolerance": 1e-7,
                                          "absolute_tolerance": 1e-8}}

# snes_solver_parameters = {"nonlinear_solver": "newton",
#                             "newton_solver": {"maximum_iterations": 50,
#                                             "report": True,
#                                             "error_on_nonconvergence": False,
#                                             'convergence_criterion': 'residual',
#                                             "relative_tolerance":1e-6,
#                                             "absolute_tolerance":1e-6}}


# comm_rank = MPI.rank(comm)
##########################################################################
##########################################################################
# Definitions
# 1) Total potential energy
def Psi(u, p, Cv):
    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)  # Identity Tensor
    F = I + grad(u)
    C = F.T * F
    Cv_inv = inv(Cv)
    I1 = tr(C)
    JF = det(F)

    Ie1 = inner(C, Cv_inv)
    Je = JF / sqrt(det(Cv))

    Jstar = (lam1 + p + sqrt((p + lam1)**2 + 4 *
             lam1 * (mu1 + mu2))) / (2 * lam1)

    psi_Eq = ((3**(1 - alph1)) / (2 * alph1)) * mu1 * (I1**alph1 - 3**alph1) +\
        ((3**(1 - alph2)) / (2 * alph2)) * mu2 * (I1**alph2 - 3**alph2) -\
        (mu1 + mu2) * ln(Jstar) + 0.5 * lam1 * (Jstar - 1)**2

    psi_Neq = ((3**(1 - a1)) / (2 * a1)) * m1 * (Ie1**a1 - 3**a1) +\
        ((3**(1 - a2)) / (2 * a2)) * m2 * (Ie1**a2 - 3**a2) -\
        (m1 + m2) * ln(Je)

    W_hat = (psi_Eq + p * (JF - Jstar) + psi_Neq) * dx

    return W_hat

# 2) Stress Piola


def stressPiola(u, p, Cv):
    d = u.geometric_dimension()
    I = Identity(d)  # Identity Tensor
    F = I + grad(u)
    Finv = inv(F)
    C = F.T * F
    Cv_inv = inv(Cv)
    I1 = tr(C)
    Iv1 = tr(Cv)

    Ie1 = inner(C, Cv_inv)
    Ie2 = 0.5 * (Ie1**2 - inner(Cv_inv * C, C * Cv_inv))

    SEq = (mu1 * (I1 / 3.)**(alph1 - 1) +
           mu2 * (I1 / 3.)**(alph2 - 1)) * F +\
        p * Finv.T
    SNeq = (m1 * (Ie1 / 3.)**(a1 - 1.) +
            m2 * (Ie1 / 3.)**(a2 - 1)) * F * Cv_inv -\
        (m1 + m2) * Finv.T

    return SEq + SNeq

# 3) Evolution Equation


def Eveq_Cv(u, Cv):
    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)  # Identity Tensor
    F = I + grad(u)
    C = F.T * F
    Cv_inv = inv(Cv)
    I1 = tr(C)
    Iv1 = tr(Cv)

    Ie1 = inner(C, Cv_inv)
    Ie2 = 0.5 * (Ie1**2 - inner(Cv_inv * C, C * Cv_inv))

    c_neq = m1 * (Ie1 / 3.)**(a1 - 1.) + m2 * (Ie1 / 3.)**(a2 - 1)
    J2Neq = (Ie1**2 / 3 - Ie2) * c_neq**2

    etaK = etaInf + (eta0 - etaInf + K1 * (Iv1**bta1 - 3.**bta1)
                     ) / (1 + (K2 * J2Neq)**bta2)

    # etaK=10000*eta0

    G = (c_neq / etaK) * (C - Ie1 * Cv / 3)
    G = local_project(G, V_Cv)

    return G

# 4) K terms


def k_t(dt, u, un, Cvn):
    un_quar = un + 0.25 * (u - un)
    un_half = un + 0.5 * (u - un)
    un_thr_quar = un + 0.75 * (u - un)
    k1 = Eveq_Cv(un, Cvn)
    k2 = Eveq_Cv(un_half, Cvn + k1 * dt / 2)
    k3 = Eveq_Cv(un_quar, Cvn + dt * (3 * k1 + k2) / 16)
    k4 = Eveq_Cv(un_half, Cvn + dt * k3 / 2.)
    k5 = Eveq_Cv(un_thr_quar, Cvn + 3 * dt * (-k2 + 2. * k3 + 3. * k4) / 16)
    k6 = Eveq_Cv(u, Cvn + (k1 + 4. * k2 + 6. *
                 k3 - 12. * k4 + 8. * k5)* dt / 7.)

    k = dt * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6) / 90

    k = local_project(k, V_Cv)
    return k

# 5) Efficient projection


metadata = {"quadrature_degree": 4}


def local_project(v, V):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_) * dx(metadata=metadata)
    b_proj = inner(v, v_) * dx(metadata=metadata)
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u


# 6) Evaluate Function (if in parallel)
def evaluate_function(u, x):
    comm = u.function_space().mesh().mpi_comm()
    if comm.size == 1:
        return u(*x)

    # Find whether the point lies on the partition of the mesh local
    # to this process, and evaulate u(x)
    cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(*x))
    u_eval = u(*x) if distance < DOLFIN_EPS else None

    # Gather the results on process 0
    comm = mesh.mpi_comm()
    computed_u = comm.gather(u_eval, root=0)

    # Verify the results on process 0 to ensure we see the same value
    # on a process boundary
    if comm.rank == 0:
        global_u_evals = np.array(
            [y for y in computed_u if y is not None], dtype=np.double)
        assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)

        computed_u = global_u_evals[0]
    else:
        computed_u = None

    # Broadcast the verified result to all processes
    computed_u = comm.bcast(computed_u, root=0)

    return computed_u

# 7) Extract Top_dofs
# def extract_dofs_boundary(V_up, bsubd):
    # label = Function(V_up)
    # bsubd_dofs = np.where(label.vector()==1)[0]
    # return bsubd_dofs


##########################################################################
##########################################################################
# Create mesh and define function space
mesh = UnitCubeMesh(3, 3, 3)
W1 = VectorElement('CG', mesh.ufl_cell(), 2)
W2 = FiniteElement('CG', mesh.ufl_cell(), 1)
W3 = TensorElement("DG", mesh.ufl_cell(), 0)  # for Cv

V = FunctionSpace(mesh, W1 * W2)
V_Cv = FunctionSpace(mesh, W3)
V_u = FunctionSpace(mesh, W1)       # for u
# V = FunctionSpace(mesh,MixedElement([W1,W2])) also works

# Define Trial and Test Function
du, dp = TrialFunctions(V)          # Incremental displacement
dw = TrialFunction(V)
v, ph = TestFunctions(V)            # Test Function
wh = TestFunction(V)
w = Function(V)                    # Displacement from previous iteration
u, p = split(w)

Cv = Function(V_Cv)
Cvn = Function(V_Cv, name='Cv')
Cv_temp = Function(V_Cv)
Cv_nk = Function(V_Cv)
un = Function(V_u, name='u')
unk = Function(V_u)
stressplot = Function(V_Cv, name='S')

Cv.assign(project(Identity(3), V_Cv))  # Initialize Cv
Cvn.assign(project(Identity(3), V_Cv))  # Initialize Cvn
un.vector()[:] = 0.                   # Initialize un
##########################################################################
##########################################################################
# Mark boundary subdomains
left_fc = CompiledSubDomain(
    "near(x[0], 0.) && on_boundary"
)
right_fc = CompiledSubDomain(
    "near(x[0], 1.) && on_boundary"
)
bottom_fc = CompiledSubDomain(
    "near(x[1], 0.) && on_boundary"
)
back_fc = CompiledSubDomain(
    "near(x[2], 0.) && on_boundary"
)
##########################################################################
##########################################################################
strtch = Constant(0.0)

bc_right = DirichletBC(V.sub(0).sub(0), strtch, right_fc)
bc_left = DirichletBC(V.sub(0).sub(0), Constant(0.0), left_fc)
bc_bottom = DirichletBC(V.sub(0).sub(1), Constant(0.0), bottom_fc)
bc_back = DirichletBC(V.sub(0).sub(2), Constant(0.0), back_fc)

bcs = [bc_right, bc_left, bc_bottom, bc_back]


##########################################################################
##########################################################################
# Material Properties VHB4910
mu1 = Constant(13.54 * 1e3)
mu2 = Constant(1.08 * 1e3)
lam1 = Constant((mu1 + mu2) * 10**3)  # make this value very high
alph1 = Constant(1.)
alph2 = Constant(-2.474)
m1 = Constant(5.42 * 1e3)
m2 = Constant(20.78 * 1e3)
a1 = Constant(-10.)
a2 = Constant(1.948)
K1 = Constant(3507 * 1e3)
K2 = Constant(1.0 * 1.e-12)
bta1 = Constant(1.852)
bta2 = Constant(0.26)
eta0 = Constant(7014 * 1e3)
etaInf = Constant(0.1 * 1e3)
##########################################################################
##########################################################################

# Set up the variational problem and solve
# solver_parameters = {'newton_solver':{'relative_tolerance':1.e-6}}
S = derivative(Psi(u, p, Cv), w, wh)
# R=derivative(Psi(u,p,Cv),u,v)
Jac = derivative(S, w, dw)
problem = NonlinearVariationalProblem(S, w, bcs, J=Jac)
solver = NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)
##########################################################################
##########################################################################
# Loading Data
lamdot = 0.01        # Loading Rate
Tfinal = 2 * (2 / lamdot)
# print('Loading Time ={}'.format(Tfinal/2))
wfil = XDMFFile('disp_visco_VHB_head.xdmf')
wfil.parameters["flush_output"] = True
wfil.parameters["functions_share_mesh"] = True
wfil.parameters["rewrite_function_mesh"] = False

wres = Function(V)
##########################################################################
##########################################################################
# Begin incremental loading
n_count = 101
dt = Tfinal / (n_count - 1)

timeVals = np.linspace(0, Tfinal, n_count)

stretchVals = np.hstack((lamdot * timeVals[:len(timeVals) // 2], lamdot *
                        (-timeVals[len(timeVals) // 2:] + 2 * timeVals[len(timeVals) // 2])))
sPiolaVals = np.zeros((timeVals.shape[0], 3))


pt = (1., 1.0, 1.0)


for i, t in enumerate(np.linspace(0, Tfinal, n_count)):
    print('Time = {}'.format(t))
    # strtch.assign(lamdot*t)
    strtch.assign(stretchVals[i])
    print('Stretch={}'.format(float(strtch)))

    solver.solve()
    u, p = w.split(True)
    # print('u before staggered={}'.format(float(evaluate_function(u, pt)[0])))

    Cv_temp.assign(Cvn)
    Cv_nk = Cvn + k_t(dt, u, un, Cvn)

    Cv_nk = local_project(Cv_nk, V_Cv)
    norm_Cv = norm(Cv_temp.vector() - Cv_nk.vector())

    # print('norm={}'.format(norm_Cv))
    iter = 0.

    while norm_Cv > 1.e-5 and iter <= 20:
        Cv_temp.assign(Cv_nk)
        Cv.assign(Cv_temp)
        solver.solve()
        u, p = w.split(True)

        unk.assign(u)

        Cv_nk = Cvn + k_t(dt, unk, un, Cvn)
        Cv_nk = local_project(Cv_nk, V_Cv)
        norm_Cv = norm(Cv_temp.vector() - Cv_nk.vector())

        # print("Inside Staggered norm={}".format(norm_Cv))
        iter = iter + 1
        # print("Iter Count={}".format(iter))

    Cvn.assign(Cv_nk)
    un.assign(unk)
    stressplot.assign(local_project(stressPiola(un, p, Cvn), V_Cv))

    sPiolaVals[i] = np.array([evaluate_function(stressplot, pt)[0], evaluate_function(
        stressplot, pt)[4], evaluate_function(stressplot, pt)[8]], float)
    print("S11 = {}".format(float(sPiolaVals[i, 0])))

    wfil.write(u, t)
    wfil.write(stressplot, t)
    wfil.write(Cvn, t)
    # wfil.write()

# Fx=MPI.sum(comm,sum(fint[top_dofs]))
# print('Fx={}'.format(Fx))
plt.figure(figsize=(8, 8))
# plt.plot(timeVals, sPiolaVals[:, 0] , label='S11')
plt.plot(1 + stretchVals, sPiolaVals[:, 0]/(float(mu1.values() + mu2.values())), label='S11')
plt.grid(True)
plt.legend(loc=0)
plt.savefig('S11_VHB_head.png')
print('Stress last = {}'.format(max(sPiolaVals[:, 0])))
np.savetxt('VHB_4910_pre.txt', np.vstack((stretchVals, sPiolaVals[:, 2])).T)
