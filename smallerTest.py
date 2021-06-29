import numpy as np
from firedrake import *

parameters["form_compiler"]["quadrature_degree"]=7

def Eveq_Cv(u, Cv):
    I = Identity(3)  # Identity Tensor
    F = I + grad(u)
    C = F.T * F
    Cv_inv = inv(Cv)
    Iv1 = tr(Cv)

    Ie1 = inner(C, Cv_inv)
    Ie2 = 0.5 * (Ie1**2. - inner(Cv_inv * C, C * Cv_inv))

    c_neq = m1 * (Ie1 / 3.)**(a1 - 1.) + m2 * (Ie1 / 3.)**(a2 - 1)
    J2Neq = (Ie1**2 / 3. - Ie2) * c_neq**2.

    etaK = etaInf + (eta0 - etaInf + K1 * (Iv1**bta1 - 3.**bta1)) / (1 + (K2 * J2Neq)**bta2)

    G = (c_neq / etaK) * (C - Ie1 * Cv / 3)
    G = project(G, V_Cv)

    print(f"Int Ie1: {assemble(Ie1*dx)}")
    print(f"Int Ie2: {assemble(Ie2*dx)}")
    print(f"Int Iv1: {assemble(Iv1*dx)}")
    print(f"Int J2: {assemble(J2Neq*dx)}")
    print(f"Int etaK: {assemble(etaK*dx)}")
    print(f"Norm G: {norm(G)}")
    return G


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


mesh = UnitCubeMesh(3, 3, 3)
W1 = VectorElement('CG', mesh.ufl_cell(), 2)
W2 = FiniteElement('CG', mesh.ufl_cell(), 1)
W3 = TensorElement("DG", mesh.ufl_cell(), 0)  # for Cv

V = FunctionSpace(mesh, W1 * W2)
V_Cv = FunctionSpace(mesh, W3)

w = Function(V)
u, p = split(w)
wh = TestFunction(V)
dw = TrialFunction(V)
Cv = Function(V_Cv)
# u.interpolate(Constant([0., 0., 0.]))
Cv.interpolate(Constant(np.eye(3)))

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

lamdot = 0.01
Tfinal = 2 * (2. / lamdot)
n_count = 101
dt = Tfinal / (n_count - 1)

timeVals = np.linspace(0, Tfinal, n_count)

stretchVals = np.hstack((lamdot * timeVals[:len(timeVals) // 2], lamdot *
                        (-timeVals[len(timeVals) // 2:] + 2 * timeVals[len(timeVals) // 2])))

left_fc = 1
right_fc = 2
bottom_fc = 3
back_fc = 5
strtch = Constant(0.0)

bc_right = DirichletBC(V.sub(0).sub(0), strtch, right_fc)
bc_left = DirichletBC(V.sub(0).sub(0), Constant(0.0), left_fc)
bc_bottom = DirichletBC(V.sub(0).sub(1), Constant(0.0), bottom_fc)
bc_back = DirichletBC(V.sub(0).sub(2), Constant(0.0), back_fc)

bcs = [bc_right, bc_left, bc_bottom, bc_back]

S = derivative(Psi(u, p, Cv), w, wh)
Jac = derivative(S, w, dw)
problem = NonlinearVariationalProblem(S, w, bcs, J=Jac)
solver_params={
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "snes_max_it": 100,
    "snes_rtol": 1.e-7,
    "snes_atol": 1.e-8,
    "snes_monitor": None,
    "snes_linesearch_type": "basic"
}
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_params)
solver.solve()
u, p = w.split()
print(
    f"||u||: {norm(u)},\
      ||p||: {norm(p)},\
      ||E||: {assemble(Psi(u, p, Cv))}"
)
Eveq_Cv(u, Cv)
