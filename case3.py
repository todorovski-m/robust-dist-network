import numpy as np

A = 4
tg = np.tan(np.arccos(0.9))
CASEDATA = {
    'Vs': 10,  # voltage at node 1 (kV)
    'Vmin': 9,  # minimum voltage(kV)
    'alpha': 0.6,  # load factor
    'cu': 4,  # cost of undelivered energy($/kWh)
    'cl': 0.1,  # cost of energy loss($/kWh)
    'g': 0.05,  # capital recovery rate
    'delta': 10,  # percentage deviation in load Smax*(1 '+/-' delta/100)
    'bus': np.array([
        # bus Pd(kW) Qd(kvar)
        [2, 250*0.9*A, 250*0.9*tg*A],
        [3, 160*0.9*A, 160*0.9*tg*A],
    ]),
    'branch': np.array([
        # number from to r(Ohm/km) x(Ohm/km) length(km) Imax(A) failure_rate
        # duration(h) cost($/km) substation_equipment($)
        [1, 1, 2, 1.2, 0.4, 2.10, 125, 0.2, 3, 15000, 75000],
        [2, 1, 3, 1.2, 0.4, 1.65, 125, 0.2, 3, 15000, 75000],
        [3, 2, 3, 2.1, 0.4, 2.00, 90, 0.2, 3, 15000, 0],
    ]),
    'xy': np.array([
        # x(mm) y(mm)
        [32.6, 5.9],
        [45.6, 17.1],
        [35.1, 25.3],
    ])
}
