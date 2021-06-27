import numpy as np
import matplotlib.pyplot as plt
from firedrake import *

parameters["form_compiler"]["quadrature_degree"] = 6
set_log_level(30)
comm = COMM_WORLD

# Definitions
# 1) Total potential energy
def Psi(u, p, Cv):
    # Kinematics
    I = Identity(3)  # Identity Tensor
    F = I + grad(u)
    C = F.T * F
    Cv_inv = inv(Cv)
    I1 = tr(C)
    JF = det(F)

    Ie1 = inner(C, Cv_inv)
    Je = JF / sqrt(det(Cv))

    Jstar = (lam1 + p + sqrt((p + lam1)**2 + 4 *\
             lam1 * (mu1 + mu2))) / (2 * lam1)

    psi_Eq = ((3**(1 - alph1)) / (2 * alph1)) * mu1 * (I1**alph1 - 3**alph1) +\
        ((3**(1 - alph2)) / (2 * alph2)) * mu2 * (I1**alph2 - 3**alph2) -\
        (mu1 + mu2) * ln(Jstar) + 0.5 * lam1 * (Jstar - 1)**2

    psi_Neq = ((3**(1 - a1)) / (2 * a1)) * m1 * (Ie1**a1 - 3**a1) +\
        ((3**(1 - a2)) / (2 * a2)) * m2 * (Ie1**a2 - 3**a2) -\
        (m1 + m2) * ln(Je)

    W_hat = (psi_Eq + p * (JF - Jstar) + psi_Neq) * dx

    return W_hat


def stressPiola(u, p, Cv):
    I = Identity(3)  # Identity Tensor
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

def Eveq_Cv(u, Cv):
    I = Identity(3)  # Identity Tensor
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

    G = (c_neq / etaK) * (C - Ie1 * Cv / 3)
    G = project(G, V_Cv)
    # print(f"Norm G: {norm(G)}")
    return G

def k_t(dt, u, un, Cvn):
    un_quar = un + 0.25 * (u - un)
    un_half = un + 0.5 * (u - un)
    un_thr_quar = un + 0.75 * (u - un)
    k1 = Eveq_Cv(un, Cvn)
    k2 = Eveq_Cv(un_half, Cvn + k1 * dt / 2)
    k3 = Eveq_Cv(un_quar, Cvn + dt * (3 * k1 + k2) / 16)
    k4 = Eveq_Cv(un_half, Cvn + dt * k3 / 2.)
    k5 = Eveq_Cv(un_thr_quar, Cvn + 3 * dt * (-k2 + 2. * k3 + 3. * k4) / 16)
    k6 = Eveq_Cv(u, Cvn + (k1 + 4. * k2 + 6. *\
                 k3 - 12. * k4 + 8. * k5)* dt / 7.)

    k = dt * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6) / 90

    k = project(k, V_Cv)
    # print(f"Norm k : {norm(k)}")
    return k

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

##########################################################################
##########################################################################
# Mark boundary subdomains
left_fc = 1
right_fc = 2
bottom_fc = 3
back_fc = 5
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
S = derivative(Psi(u, p, Cv), w, wh)
Jac = derivative(S, w, dw)
problem = NonlinearVariationalProblem(S, w, bcs, J=Jac)

solver_params={
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "snes_max_it": 100,
    # "snes_stol": 1.e-6,
    # "snes_rtol": 1.e-6,
    # "snes_atol": 1.e-6,
    "snes_view": None,
    # "ksp_atol": 1.e-10,
    # "ksp_rtol":1.e-8,
    "ksp_view": None,
    "snes_linesearch_type": "basic",
    "snes_no_convergence_test": 1
}
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_params)
##########################################################################
##########################################################################
# Loading Data
lamdot = 0.01        # Loading Rate
Tfinal = 2 * (2 / lamdot)
# print('Loading Time ={}'.format(Tfinal/2))
wfil = File('disp_visco_VHB_head.pvd')
wfil.write(un, Cvn, stressplot, time=0)
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


pt = [1.0, 1.0, 1.0]

for i, t in enumerate(np.linspace(0, Tfinal, n_count)[1:], start=1):
    print('Time = {}'.format(t))
    strtch.assign(stretchVals[i])
    print('Stretch={}'.format(float(strtch)))
    try:
        solver.solve()
    except:
        print("solver diverged, still continuing")
    unk, pk = w.split()
    # print('u before staggered={}'.format(float(evaluate_function(u, pt)[0])))

    Cv_temp.assign(Cvn)
    Cv_nk = Cvn + k_t(dt, unk, un, Cvn)

    Cv_nk = project(Cv_nk, V_Cv)
    norm_Cv = errornorm(Cv_temp, Cv_nk)

    print(f"norm={norm_Cv}")
    iter = 0
    while norm_Cv > 1.e-5 and iter <= 20:
        Cv_temp.assign(Cv_nk)
        Cv.assign(Cv_temp)
        try:
            solver.solve()
        except:
            print("inner (staggered) solve failed")
        unk, pnk = w.split()

        Cv_nk = Cvn + k_t(dt, unk, un, Cvn)
        Cv_nk = project(Cv_nk, V_Cv)
        norm_Cv = errornorm(Cv_temp, Cv_nk)

        print(f"Inside Staggered norm={norm_Cv}")
        iter = iter + 1
        print(f"Iter Count={iter}")

    Cvn.assign(Cv_nk)
    un.assign(unk)
    stressplot.assign(project(stressPiola(un, p, Cvn), V_Cv))

    S_at_pt = stressplot.at(pt)
    sPiolaVals[i] = np.array([S_at_pt[0, 0], S_at_pt[1, 1], S_at_pt[2, 2]], float)
    print("S11 = {}".format(float(sPiolaVals[i, 0])))

    wfil.write(un, Cvn, stressplot, time=t)

plt.figure(figsize=(8, 8))
plt.plot(1 + stretchVals, sPiolaVals[:, 0]/(float(mu1.values() + mu2.values())), label='S11')
plt.grid(True)
plt.legend(loc=0)
plt.savefig('S11_VHB_head.png')
print('Stress last = {}'.format(max(sPiolaVals[:, 0])))
np.savetxt('VHB_4910_pre.txt', np.vstack((stretchVals, sPiolaVals[:, 2])).T)
