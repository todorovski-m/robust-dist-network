import importlib
import numpy as np
import networkx as nx
import pyomo.environ as pyo
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def solve_model(o, m):
    if hasattr(o, 'set_instance'):
        o.set_instance(m)  # required by gurobi
    r = o.solve(m, tee=False)
    return r


def load_case(case_file):
    # citanje na vleznata datoteka
    print(f'\ncase_file = {case_file}')
    case_file = case_file.replace('.py', '')
    my_module = importlib.import_module(case_file)
    ds = getattr(my_module, 'CASEDATA')

    # podatoci za jazlite
    nb = ds['xy'].shape[0]
    ibus, P1, Q1 = ds['bus'].T
    ibus = ibus.astype(int) - 1
    Pd = np.zeros(nb)
    Qd = np.zeros(nb)
    Pd[ibus] = P1/1000
    Qd[ibus] = Q1/1000
    delta = ds['delta']

    # podatoci za mrezata
    nl = ds['branch'].shape[0]
    _, f, t, R, X, d, Imax, fail_rate, duration, ck, sub_equip = ds['branch'].T
    f = f.astype(int) - 1
    t = t.astype(int) - 1
    R = R * d
    X = X * d
    ck = ck * d + sub_equip
    Imax = Imax/1000
    lam = fail_rate * d
    beta = 0.15*ds['alpha'] + 0.85*ds['alpha']**2

    # graf na site granki na mrezata
    G = nx.Graph()
    for i, p in enumerate(ds['xy']):
        G.add_node(i, pos=tuple(p))
    for i, j in zip(f, t):
        G.add_edge(i, j)

    # matrica na incidencija
    Af = csr_matrix((np.ones(nl), (f, range(nl))), shape=(nb, nl))
    At = csr_matrix((np.ones(nl), (t, range(nl))), shape=(nb, nl))
    A = Af - At

    # pomosni vektori za funkcijata na cel i za ogranicuvanjata
    C = ds['g'] * ck
    LAM = ds['cu'] * ds['alpha'] * (duration * lam) * 1000
    L = 8760 * ds['cl'] * beta * R / ds['Vs']**2*1000
    Pmin = Pd*(1 - delta/100)
    Pmax = Pd*(1 + delta/100)
    Smax = 3**0.5*ds['Vs']*Imax

    data = {
        'Vs': ds['Vs'],
        'Vmin': ds['Vmin'],
        'f': f,
        't': t,
        'A': A,
        'C': C,
        'LAM': LAM,
        'L': L,
        'R': R,
        'X': X,
        'Pd': Pd,
        'Qd': Qd,
        'Pmin': Pmin,
        'Pmax': Pmax,
        'Smax': Smax,
        'G': G,
        'beta': beta,
        'cl': ds['cl'],
    }

    return data


def deterministic(data):
    nb, nl = data['A'].shape
    DP = pyo.ConcreteModel(name='Deterministic model')

    # indices
    DP.lines = pyo.Set(initialize=range(nl))
    DP.buses = pyo.Set(initialize=range(nb))
    DP.loads = pyo.Set(initialize=range(1, nb))

    # line status
    DP.b = pyo.Var(DP.lines, within=pyo.Binary)
    # line active power flow
    DP.P = pyo.Var(DP.lines)
    # line reactive power flow
    DP.Q = pyo.Var(DP.lines)
    # square of bus voltage
    DP.W = pyo.Var(DP.buses, bounds=(data['Vmin']**2, None))
    # difference of square of voltages at both line ends
    DP.U = pyo.Var(DP.lines, bounds=(-data['Vs']**2, data['Vs']**2))
    # variable to linearize b*U
    DP.F = pyo.Var(DP.lines)
    # variable to linearize abs(P)
    DP.H = pyo.Var(DP.lines, within=pyo.NonNegativeReals)

    # line cost ($)
    DP.Cc = sum(data['C'][i]*DP.b[i] for i in DP.lines)
    # supply interruption cost ($)
    DP.Ce = sum(data['LAM'][i]*DP.H[i] for i in DP.lines)
    # cost of energy losses ($)
    DP.Cl = sum(data['L'][i]*(DP.P[i]**2 + DP.Q[i]**2) for i in DP.lines)
    # objective
    DP.obj = pyo.Objective(expr=DP.Cc + DP.Ce + DP.Cl)

    # radial network sum(b) == nb - 1
    DP.radial = pyo.Constraint(expr=sum(DP.b[i] for i in DP.lines) == nb - 1)

    # supply bus voltage
    DP.supply = pyo.Constraint(expr=DP.W[0] == data['Vs']**2)

    # line voltage equation U = A' * W
    DP.line_voltage = pyo.ConstraintList()
    for i in DP.lines:
        DP.line_voltage.add(expr=DP.U[i] == sum(
            data['A'][j, i]*DP.W[j] for j in DP.buses))

    # line power flow F == 2*(P*R + Q*X)
    DP.line_flow = pyo.ConstraintList()
    for i in DP.lines:
        DP.line_flow.add(expr=DP.F[i] == 2*(DP.P[i]*data['R'][i] +
                                            DP.Q[i]*data['X'][i]))

    # limits on line power flows
    DP.line_limit = pyo.ConstraintList()
    # limits on active power flow
    for i in DP.lines:
        DP.line_limit.add(DP.P[i] >= -data['Smax'][i]*DP.b[i])
        DP.line_limit.add(DP.P[i] <= data['Smax'][i]*DP.b[i])
    # limits on reactive power flow
    for i in DP.lines:
        DP.line_limit.add(DP.Q[i] >= -data['Smax'][i]*DP.b[i])
        DP.line_limit.add(DP.Q[i] <= data['Smax'][i]*DP.b[i])

    # limits on apparent power flow
    for i in DP.lines:
        DP.line_limit.add(DP.P[i]**2 + DP.Q[i]**2 <= data['Smax'][i]**2)

    # load balance for active power A * P == -Pd
    DP.load_balance = pyo.ConstraintList()
    for i in DP.loads:
        DP.load_balance.add(expr=sum(data['A'][i, j]*DP.P[j]
                                     for j in DP.lines) == -data['Pd'][i])
    # load balance for reactive power A * Q == -Qd
    for i in DP.loads:
        DP.load_balance.add(expr=sum(data['A'][i, j]*DP.Q[j]
                                     for j in DP.lines) == -data['Qd'][i])

    # linearization F = b*U
    # b is binary, min <= U <= max (-Vs^2 <= U <= Vs^2)
    # min*b <= F <= max*b
    # U - max*(1-b) <= F <= U - min*(1-b)
    DP.linear = pyo.ConstraintList()
    Vs = data['Vs']
    for i in DP.lines:
        DP.linear.add(DP.F[i] >= -Vs**2*DP.b[i])
        DP.linear.add(DP.F[i] <= Vs**2*DP.b[i])
        DP.linear.add(DP.F[i] >= DP.U[i] + Vs ** 2*DP.b[i] - Vs**2)
        DP.linear.add(DP.F[i] <= DP.U[i] - Vs ** 2*DP.b[i] + Vs**2)

    # linearization H = abs(P)
    DP.abs = pyo.ConstraintList()
    for i in DP.lines:
        DP.abs.add(DP.H[i] >= -DP.P[i])
        DP.abs.add(DP.H[i] >= DP.P[i])

    return DP


def subproblem(data, gama):
    nb, nl = data['A'].shape
    SP = pyo.ConcreteModel(name='Subproblem')

    # indices
    SP.lines = pyo.Set(initialize=range(nl))
    SP.buses = pyo.Set(initialize=range(nb))
    SP.loads = pyo.Set(initialize=range(1, nb))

    def line_bound(SP, i):
        return (-data['Smax'][i], data['Smax'][i])
    # line status
    SP.b = pyo.Param(SP.lines, mutable=True)
    # active load demand
    SP.Pd = pyo.Var(SP.loads)
    # reactive load demand
    SP.Qd = pyo.Var(SP.loads)
    # linearize abs(Pd - Pd^{ref}) using new variable t
    SP.t = pyo.Var(SP.loads)
    # line active power flow
    SP.P = pyo.Var(SP.lines, bounds=line_bound)
    # line reactive power flow
    SP.Q = pyo.Var(SP.lines, bounds=line_bound)
    # square of bus voltage
    SP.W = pyo.Var(SP.buses, bounds=(data['Vmin']**2, None))
    # difference of square of voltages at both line ends
    SP.U = pyo.Var(SP.lines, bounds=(-data['Vs']**2, data['Vs']**2))
    # variables to linearize abs(P)
    SP.H = pyo.Var(SP.lines, within=pyo.NonNegativeReals)
    SP.B = pyo.Var(SP.lines, within=pyo.Binary)

    # objective
    SP.Ce = sum(data['LAM'][i]*SP.H[i] for i in SP.lines)
    SP.Cl = sum(data['L'][i]*(SP.P[i]**2 + SP.Q[i]**2) for i in SP.lines)
    SP.obj = pyo.Objective(expr=SP.Ce + SP.Cl, sense=pyo.maximize)

    # supply bus voltage
    SP.supply = pyo.Constraint(expr=SP.W[0] == data['Vs']**2)

    # line voltage equation U = A' * W
    SP.line_voltage = pyo.ConstraintList()
    for i in SP.lines:
        SP.line_voltage.add(expr=SP.U[i] == sum(
            data['A'][j, i]*SP.W[j] for j in SP.buses))

    # line power flow b*U == 2*(P*R + Q*X) (b = const)
    SP.line_flow = pyo.ConstraintList()
    for i in SP.lines:
        SP.line_flow.add(expr=SP.b[i]*SP.U[i] == 2*(SP.P[i]*data['R'][i] +
                                                    SP.Q[i]*data['X'][i]))
    # limits on line power flows
    SP.limit_pq = pyo.ConstraintList()
    # limits on active power flow
    for i in SP.lines:
        SP.limit_pq.add(SP.P[i] >= -data['Smax'][i]*SP.b[i])
        SP.limit_pq.add(SP.P[i] <= data['Smax'][i]*SP.b[i])
    # limits on reactive power flow
    for i in SP.lines:
        SP.limit_pq.add(SP.Q[i] >= -data['Smax'][i]*SP.b[i])
        SP.limit_pq.add(SP.Q[i] <= data['Smax'][i]*SP.b[i])

    # limits on apparent power flow with quadratic terms
    # to be used with GUROBI
    SP.limit_s_quad = pyo.ConstraintList()
    for i in SP.lines:
        SP.limit_s_quad.add(SP.P[i]**2 + SP.Q[i]**2 <= data['Smax'][i]**2)

    def parabola(SP, j, x):
        # we not need j, but it is passed as the index for the constraint
        return x**2
    # limits on apparent power flow with piecewise linear terms
    # to be used with CPLEX
    SP.limit_s_pwl = pyo.ConstraintList()
    # pwl variable for active power flow
    SP.P2 = pyo.Var(SP.lines)
    # pwl revariable for active power flow
    SP.Q2 = pyo.Var(SP.lines)
    # interpolation points
    pts = {}
    for i in SP.lines:
        s = data['Smax'][i]
        pts[i] = np.linspace(-s, s, 10, endpoint=True).tolist()
    # constraints replacing P with pwl approximation
    SP.pwl_p2 = pyo.Piecewise(
        SP.lines, SP.P2, SP.P, pw_pts=pts, pw_constr_type='EQ', f_rule=parabola)
    # constraints replacing Q with pwl approximation
    SP.pwl_q2 = pyo.Piecewise(
        SP.lines, SP.Q2, SP.Q, pw_pts=pts, pw_constr_type='EQ', f_rule=parabola)
    for i in SP.lines:
        # SP.line_limit.add(SP.P[i]**2 + SP.Q[i]**2 <= data['Smax'][i]**2)
        SP.limit_s_pwl.add(SP.P2[i] + SP.Q2[i] <= data['Smax'][i]**2)

    # load balance for active power A * P == -Pd
    SP.load_balance = pyo.ConstraintList()
    for i in SP.loads:
        SP.load_balance.add(expr=sum(data['A'][i, j]*SP.P[j]
                                     for j in SP.lines) == -SP.Pd[i])
    # load balance for reactive power A * Q == -Qd
    for i in SP.loads:
        SP.load_balance.add(expr=sum(data['A'][i, j]*SP.Q[j]
                                     for j in SP.lines) == -SP.Qd[i])

    # linearization H = abs(P)
    # binary variable B and continuous variable H
    # constraints to the model:
    #  P + M * B >= H
    # -P + M * (1 - B) >= H
    #  H >=  P
    #  H >= -P
    # maximum value needed for M for this to work is M = 2*Smax
    M = 2*data['Smax']
    SP.abs = pyo.ConstraintList()
    for i in SP.lines:
        SP.abs.add(SP.P[i] + M[i]*SP.B[i] >= SP.H[i])
        SP.abs.add(-SP.P[i] + M[i]*(1 - SP.B[i]) >= SP.H[i])
        SP.abs.add(SP.H[i] >= SP.P[i])
        SP.abs.add(SP.H[i] >= -SP.P[i])

    # robust set
    SP.box = pyo.ConstraintList()
    for i in SP.loads:
        SP.box.add(SP.Pd[i] >= data['Pmin'][i])
        SP.box.add(SP.Pd[i] <= data['Pmax'][i])
        SP.box.add(SP.Qd[i] == data['Qd'][i]/data['Pd'][i]*SP.Pd[i])
    # linerize abs(D - D^{ref}) using new variable t
    # Pd^{ref} = data['Pd']
    # Pd^{Delta} = data['Pd'] - data['Pmin']
    SP.absd = pyo.ConstraintList()
    for i in SP.loads:
        SP.absd.add(SP.t[i] >= SP.Pd[i] - data['Pd'][i])
        SP.absd.add(SP.t[i] >= data['Pd'][i] - SP.Pd[i])
    SP.gama = pyo.Constraint(expr=sum(SP.t[i] for i in SP.loads) <=
                             gama*sum(data['Pd'][i]-data['Pmin'][i] for i in SP.loads))

    # line cost ($)
    SP.Cc = sum(data['C'][i]*SP.b[i] for i in SP.lines)

    return SP


def master_problem(data):
    nb, nl = data['A'].shape
    MP = pyo.ConcreteModel(name='Master problem')

    # set of indices
    MP.it = pyo.Set(initialize=[1])
    MP.lines = pyo.Set(initialize=range(nl))
    MP.buses = pyo.Set(initialize=range(nb))
    MP.loads = pyo.Set(initialize=range(1, nb))

    # active load demand
    MP.Pd = pyo.Param(MP.loads, MP.it, mutable=True)
    # reactive load demand
    MP.Qd = pyo.Param(MP.loads, MP.it, mutable=True)
    # line status
    MP.b = pyo.Var(MP.lines, within=pyo.Binary)
    # link with the subproblem
    MP.eta = pyo.Var(within=pyo.NonNegativeReals)
    # line active power flow
    MP.P = pyo.Var(MP.lines, MP.it)
    # line reactive power flow
    MP.Q = pyo.Var(MP.lines, MP.it)
    # square of bus voltage
    MP.W = pyo.Var(MP.buses, MP.it, bounds=(data['Vmin']**2, None))
    # difference of square of voltages at both line ends
    MP.U = pyo.Var(MP.lines, MP.it, bounds=(-data['Vs']**2, data['Vs']**2))
    # variable to linearize b*U
    MP.F = pyo.Var(MP.lines, MP.it)
    # variable to linearize abs(P)
    MP.H = pyo.Var(MP.lines, MP.it, within=pyo.NonNegativeReals)

    # objective
    MP.Cc = sum(data['C'][i]*MP.b[i] for i in MP.lines)
    MP.obj = pyo.Objective(expr=MP.Cc + MP.eta)

    # radial network sum(b) == nb - 1
    MP.radial = pyo.Constraint(expr=sum(MP.b[i] for i in MP.lines) == nb - 1)
    # limit on eta
    MP.eta_limit = pyo.ConstraintList()
    # linearization H = abs(P)
    MP.abs = pyo.ConstraintList()
    # supply bus voltage
    MP.supply = pyo.ConstraintList()
    # line voltage equation U = A' * W
    MP.line_voltage = pyo.ConstraintList()
    # line power flow F == 2*(P*R + Q*X)
    MP.line_flow = pyo.ConstraintList()
    # limits on line power flows and apparent power flow
    MP.line_limit = pyo.ConstraintList()
    # load balance for active and reactive power A * P == -Pd, A * Q == -Qd
    MP.load_balance = pyo.ConstraintList()
    # linearization F = b*U
    MP.linear = pyo.ConstraintList()

    return MP


def optimal_graph(data, b):
    T = nx.Graph()
    for u, d in data['G'].nodes(data=True):
        T.add_node(u, pos=d['pos'])
    for i, j, k in zip(data['f'], data['t'], b):
        if k == 1:
            T.add_edge(i, j)
    return T


def plot_graphs(G, T):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, edge_color='#C0C0C0', node_size=0, style='dashed')
    pos = nx.get_node_attributes(T, 'pos')
    nx.draw(T, pos, edge_color='k', node_size=0, width=4)
    buses = nx.draw_networkx_nodes(T, pos, node_color='#C0C0C0')
    buses.set_edgecolor('k')
    labels = {i: i+1 for i in range(len(T))}
    nx.draw_networkx_labels(T, pos, labels=labels)
    plt.show()
