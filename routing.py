import sys
import time
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from models import load_case
from models import deterministic
from models import subproblem
from models import solve_model
from models import master_problem


def robust_optimiation(data, gama=1):
    # log file
    flog = open('routing.log', 'w')
    flog.write(f'case_file: {CASE_FILE}\n')

    # define models
    DP = deterministic(data)
    SP = subproblem(data, gama)
    MP = master_problem(data)

    # solve the deterministic model
    opt = SolverFactory('cplex')
    t0 = time.time()
    opt.solve(DP, tee=False)
    t = time.time() - t0
    print(f'deterministic solved, t = {t:.2f} sec')
    flog.write(f'\ndeterministic, DP.obj = {pyo.value(DP.obj):.2f}\n')
    flog.write(f'deterministic, DP.Cc  = {pyo.value(DP.Cc):.2f}\n')
    flog.write(f'deterministic, DP.Ce  = {pyo.value(DP.Ce):.2f}\n')
    flog.write(f'deterministic, DP.Cl  = {pyo.value(DP.Cl):.2f}\n')
    b_det = [round(pyo.value(DP.b[i])) for i in DP.lines]
    flog.write(f'deterministic, b_det  = {b_det}\n')
    losses = pyo.value(DP.Cl)/(8760*data['cl']*data['beta'])
    Pd = sum(data['Pd'])*1000
    flog.write(f'deterministic, losses = {losses:.3f} kW\n')
    flog.write(f'deterministic, Pd = {Pd:.3f} kW\n')
    flog.write(f'deterministic, losses = {losses/(Pd + losses)*100:.2f} %\n')
    V = [pyo.value(DP.W[i])**0.5 for i in DP.buses]
    flog.write(f'deterministic, Vmin = {min(V):.3f} kV\n')
    flog.write(
        f'deterministic, dVmax = {(data["Vs"]-min(V))/data["Vs"]*100:.2f} %\n')
    flog.write(f't = {t:.2f} sec\n')

    flog.write(f'\ngama  = {gama:.2f}\n')

    # iterations
    it = 0
    UB, LB = 1e10, -1e10
    eps = 0.01
    while abs(UB - LB) > eps and it <= 10:
        it = it + 1
        print(f'\niteration = {it}')

        if it == 1:
            # take branch status from the solution of the deterministic model
            for i in SP.lines:
                SP.b[i] = pyo.value(DP.b[i])
        else:
            # get branch status from the solution of the master problem and use it in
            # the subproblem
            print('# SP.b[i] = round(pyo.value(MP.b[i]))')
            for i in SP.lines:
                SP.b[i] = round(pyo.value(MP.b[i]))

        # solve the subproblem
        solver = 'cplex'
        # solver = 'gurobi_persistent'
        opt = SolverFactory(solver)
        if solver == 'cplex':
            opt.options['optimalitytarget'] = 3
            SP.limit_s_quad.deactivate()
        else:
            opt.options['NonConvex'] = 2
            SP.limit_s_pwl.deactivate()
            SP.pwl_p2.deactivate()
            SP.pwl_q2.deactivate()
        t0 = time.time()
        solve_model(opt, SP)
        # UB = min(UB, pyo.value(SP.Cc + SP.obj))
        UB = pyo.value(SP.Cc + SP.obj)
        t = time.time() - t0
        print(f'subproblem solved, t = {t:.2f} sec')
        flog.write(f'\niter = {it}, SP.obj = {pyo.value(SP.obj):.2f}\n')
        flog.write(f'iter = {it}, SP.Cc = {pyo.value(SP.Cc):.2f}\n')
        flog.write(f'iter = {it}, SP.Ce = {pyo.value(SP.Ce):.2f}\n')
        flog.write(f'iter = {it}, SP.Cl = {pyo.value(SP.Cl):.2f}\n')
        flog.write(f'iter = {it}, UB = {UB:.2f}\n')
        Pd = [round(pyo.value(SP.Pd[i])*1000, 2) for i in SP.loads]
        Pd_rel = [round(Pd[i-1]/(data['Pmax'][i]*1000), 2) for i in SP.loads]
        flog.write(f'iter = {it}, Pd = {Pd}\n')
        flog.write(f'iter = {it}, Pd/Pmax = {Pd_rel}\n')
        flog.write(f't = {t:.2f} sec\n')

        # add new variables and constraints (C&CG)
        if it > 1:
            # for it == 1 the variables are added when MP is created
            MP.it.add(it)
            for i in MP.lines:
                MP.P.add((i, it))
                MP.Q.add((i, it))
                MP.U.add((i, it))
                MP.F.add((i, it))
            for i in MP.buses:
                MP.W.add((i, it))

        # get load demand from the subproblem and use it in the master problem
        for i in MP.loads:
            MP.Pd[i, it] = SP.Pd[i]
            MP.Qd[i, it] = SP.Qd[i]

        # add limit on eta using optimal value from the subproblem
        MP.eta_limit.add(expr=MP.eta >=
                         sum(data['LAM'][i]*MP.H[i, it] + data['L'][i] *
                             (MP.P[i, it]**2 + MP.Q[i, it]**2) for i in
                             MP.lines))
        # linearization H = abs(P)
        for i in MP.lines:
            MP.abs.add(MP.H[i, it] >= MP.P[i, it])
            MP.abs.add(MP.H[i, it] >= -MP.P[i, it])

        # supply bus voltage
        MP.supply.add(expr=MP.W[0, it] == data['Vs']**2)

        # line voltage equation U = A' * W
        for i in MP.lines:
            MP.line_voltage.add(expr=MP.U[i, it] == sum(
                data['A'][j, i]*MP.W[j, it] for j in MP.buses))

        # line power flow F == 2*(P*R + Q*X)
        for i in MP.lines:
            MP.line_flow.add(expr=MP.F[i, it] == 2*(MP.P[i, it]*data['R'][i] +
                                                    MP.Q[i, it]*data['X'][i]))

        # limits on active power flow
        for i in MP.lines:
            MP.line_limit.add(MP.P[i, it] >= -data['Smax'][i]*MP.b[i])
            MP.line_limit.add(MP.P[i, it] <= data['Smax'][i]*MP.b[i])
        # limits on reactive power flow
        for i in MP.lines:
            MP.line_limit.add(MP.Q[i, it] >= -data['Smax'][i]*MP.b[i])
            MP.line_limit.add(MP.Q[i, it] <= data['Smax'][i]*MP.b[i])
        # limits on apparent power flow
        for i in MP.lines:
            MP.line_limit.add(MP.P[i, it]**2 + MP.Q[i, it]**2 <=
                              data['Smax'][i]**2)

        # load balance for active power A * P == -Pd
        # DP.load_balance = pyo.ConstraintList()
        for i in MP.loads:
            MP.load_balance.add(expr=sum(data['A'][i, j]*MP.P[j, it]
                                         for j in MP.lines) == -MP.Pd[i, it])
        # load balance for reactive power A * Q == -Qd
        for i in MP.loads:
            MP.load_balance.add(expr=sum(data['A'][i, j]*MP.Q[j, it]
                                         for j in MP.lines) == -MP.Qd[i, it])

        # linearization F = b*U
        # b is binary, min <= U <= max (-Vs^2 <= U <= Vs^2)
        # min*b <= F <= max*b
        # U - max*(1-b) <= F <= U - min*(1-b)
        Vs = data['Vs']
        for i in MP.lines:
            MP.linear.add(MP.F[i, it] >= -Vs**2*MP.b[i])
            MP.linear.add(MP.F[i, it] <= Vs**2*MP.b[i])
            MP.linear.add(MP.F[i, it] >= MP.U[i, it] + Vs**2*MP.b[i] - Vs**2)
            MP.linear.add(MP.F[i, it] <= MP.U[i, it] - Vs**2*MP.b[i] + Vs**2)

        # solve the master problem
        opt = SolverFactory('cplex')
        t0 = time.time()
        opt.solve(MP, tee=False)
        LB = pyo.value(MP.obj)
        t = time.time() - t0
        print(f'master problem solved, t = {t:.2f} sec')
        flog.write(f'\niter = {it}, MP.Cc = {pyo.value(MP.Cc):.2f}\n')
        flog.write(f'iter = {it}, MP.eta = {pyo.value(MP.eta):.2f}\n')
        flog.write(f'iter = {it}, MP.obj = {pyo.value(MP.obj):.2f}\n')
        flog.write(f'iter = {it}, LB = {LB:.2f}\n')
        b = [round(pyo.value(MP.b[i])) for i in MP.lines]
        flog.write(f'iter = {it}, b = {b}\n')
        flog.write(f't = {t:.2f} sec\n')

    flog.write('\ndifference robust/deterministic\n')
    obj_diff = (LB/pyo.value(DP.obj) - 1)*100
    flog.write(f'(MP.obj/DP.obj - 1)={obj_diff: .2f} %\n')
    flog.write(f'b_diff = {[b_det[i] - b[i] for i in DP.lines]}\n')
    flog.close()


if __name__ == '__main__':

    if len(sys.argv) == 1:
        CASE_FILE = 'case3.py'
    else:
        CASE_FILE = sys.argv[1]

    DATA = load_case(CASE_FILE)

    robust_optimiation(DATA, 1)
    # plot_graphs(G, T)
