import numpy as np

tg = np.tan(np.arccos(0.9))
CASEDATA = {
    'Vs': 10,  # voltage at node 1 (kV)
    'Vmin': 9.5,  # minimum voltage(kV)
    'alpha': 0.6,  # load factor
    'cu': 4,  # cost of undelivered energy($/kWh)
    'cl': 0.1,  # cost of energy loss($/kWh)
    'g': 0.05,  # capital recovery rate
    'delta': 50,  # percentage deviation in load Smax*(1 '+/-' delta/100)
    'bus': np.array([
        # bus Pd(kW) Qd(kvar)
        [2, 225.0, 225.0*tg],
        [3, 144.0, 144.0*tg],
        [4, 90.0, 90.0*tg],
        [5, 90.0, 90.0*tg],
        [6, 45.0, 45.0*tg],
        [7, 90.0, 90.0*tg],
        [8, 90.0, 90.0*tg],
        [9, 225.0, 225.0*tg],
        [10, 144.0, 144.0*tg],
        [11, 90.0, 90.0*tg],
        [12, 144.0, 144.0*tg],
        [13, 90.0, 90.0*tg],
        [14, 90.0, 90.0*tg],
        [15, 90.0, 90.0*tg],
        [16, 135.0, 135.0*tg],
        [17, 72.0, 72.0*tg],
        [18, 36.0, 36.0*tg],
        [19, 90.0, 90.0*tg],
        [20, 36.0, 36.0*tg],
        [21, 54.0, 54.0*tg],
        [22, 36.0, 36.0*tg],
        [23, 72.0, 72.0*tg],
        [24, 90.0, 90.0*tg],
        [25, 27.0, 27.0*tg],
    ]),
    'branch': np.array([
        # number from to r(Ohm/km) x(Ohm/km) length(km) Imax(A) failure_rate
        # duration(h) cost($/km) substation_equipment($)
        [1, 1, 2, 1.2, 0.4, 2.10, 125, 0.2, 3, 15000, 75000],
        [2, 1, 3, 1.2, 0.4, 1.65, 125, 0.2, 3, 15000, 75000],
        [3, 1, 4, 1.2, 0.4, 2.20, 125, 0.2, 3, 15000, 75000],
        [4, 2, 5, 2.1, 0.4, 2.00, 90, 0.2, 3, 15000, 0],
        [5, 2, 6, 2.1, 0.4, 1.50, 90, 0.2, 3, 15000, 0],
        [6, 3, 6, 2.1, 0.4, 1.75, 90, 0.2, 3, 15000, 0],
        [7, 3, 7, 2.1, 0.4, 1.75, 90, 0.2, 3, 15000, 0],
        [8, 4, 7, 2.1, 0.4, 1.75, 90, 0.2, 3, 15000, 0],
        [9, 4, 8, 2.1, 0.4, 1.00, 90, 0.2, 3, 15000, 0],
        [10, 4, 12, 2.1, 0.4, 1.00, 90, 0.2, 3, 15000, 0],
        [11, 5, 9, 2.1, 0.4, 1.25, 90, 0.2, 3, 15000, 0],
        [12, 6, 9, 2.1, 0.4, 1.50, 90, 0.2, 3, 15000, 0],
        [13, 6, 10, 2.1, 0.4, 1.75, 90, 0.2, 3, 15000, 0],
        [14, 7, 10, 2.1, 0.4, 2.00, 90, 0.2, 3, 15000, 0],
        [15, 7, 11, 2.1, 0.4, 2.00, 90, 0.2, 3, 15000, 0],
        [16, 7, 8, 2.1, 0.4, 1.75, 90, 0.2, 3, 15000, 0],
        [17, 9, 15, 2.1, 0.4, 1.25, 90, 0.2, 3, 15000, 0],
        [18, 9, 10, 2.1, 0.4, 1.75, 90, 0.2, 3, 15000, 0],
        [19, 10, 14, 2.1, 0.4, 1.75, 90, 0.2, 3, 15000, 0],
        [20, 10, 13, 2.1, 0.4, 2.75, 90, 0.2, 3, 15000, 0],
        [21, 11, 13, 2.1, 0.4, 1.75, 90, 0.2, 3, 15000, 0],
        [22, 1, 16, 1.2, 0.4, 1.50, 125, 0.2, 3, 15000, 75000],
        [23, 2, 16, 2.1, 0.4, 1.05, 90, 0.2, 3, 15000, 0],
        [24, 16, 17, 2.1, 0.4, 0.75, 90, 0.2, 3, 15000, 0],
        [25, 2, 17, 2.1, 0.4, 1.05, 90, 0.2, 3, 15000, 0],
        [26, 5, 17, 2.1, 0.4, 1.00, 90, 0.2, 3, 15000, 0],
        [27, 17, 18, 2.1, 0.4, 1.50, 90, 0.2, 3, 15000, 0],
        [28, 5, 18, 2.1, 0.4, 0.75, 90, 0.2, 3, 15000, 0],
        [29, 15, 18, 2.1, 0.4, 1.25, 90, 0.2, 3, 15000, 0],
        [30, 1, 19, 1.2, 0.4, 1.55, 125, 0.2, 3, 15000, 75000],
        [31, 4, 19, 2.1, 0.4, 1.00, 90, 0.2, 3, 15000, 0],
        [32, 19, 20, 2.1, 0.4, 0.75, 90, 0.2, 3, 15000, 0],
        [33, 12, 20, 2.1, 0.4, 0.75, 90, 0.2, 3, 15000, 0],
        [34, 12, 21, 2.1, 0.4, 0.50, 90, 0.2, 3, 15000, 0],
        [35, 21, 22, 2.1, 0.4, 0.50, 90, 0.2, 3, 15000, 0],
        [36, 8, 23, 2.1, 0.4, 1.05, 90, 0.2, 3, 15000, 0],
        [37, 11, 23, 2.1, 0.4, 0.50, 90, 0.2, 3, 15000, 0],
        [38, 8, 22, 2.1, 0.4, 0.65, 90, 0.2, 3, 15000, 0],
        [39, 3, 24, 2.1, 0.4, 0.75, 90, 0.2, 3, 15000, 0],
        [40, 9, 25, 2.1, 0.4, 0.45, 90, 0.2, 3, 15000, 0],
        [41, 14, 25, 2.1, 0.4, 0.50, 90, 0.2, 3, 15000, 0],
        [42, 4, 24, 2.1, 0.4, 0.40, 90, 0.2, 3, 15000, 0],
    ]),
    'xy': np.array([
        # x(mm) y(mm)
        [32.6, 5.9],
        [45.6, 17.1],
        [35.1, 25.3],
        [23.2, 30.1],
        [69.2, 23.3],
        [55.6, 33.5],
        [35.7, 46.3],
        [17.0, 40.2],
        [70.9, 38.2],
        [57.7, 53.6],
        [14.8, 60.1],
        [10.1, 33.8],
        [31.3, 70.5],
        [77.8, 48.6],
        [84.5, 29.4],
        [51.2, 9.0],
        [58.7, 15.3],
        [76.4, 16.8],
        [17.0, 18.0],
        [13.6, 25.6],
        [5.3, 38.9],
        [10.6, 43.1],
        [15.0, 51.7],
        [28.1, 30.1],
        [74.2, 43.8],
    ])
}
