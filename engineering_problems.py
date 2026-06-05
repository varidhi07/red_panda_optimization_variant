"""
engineering_problems.py
=======================
Five classic engineering optimisation problems used in metaheuristic papers.
Each returns (name, func, lb, ub, dim) with penalty-based constraint handling.

Problems:
  1. Tension/compression spring design
  2. Welded beam design
  3. Pressure vessel design
  4. Speed reducer design
  5. Three-bar truss design
"""

import numpy as np


def _penalty(violations, coeff=1e6):
    """Sum of squared constraint violations × large coefficient."""
    return coeff * sum(max(0, v)**2 for v in violations)


# ── 1. Tension / Compression Spring ─────────────────────────────────────────────

def spring_func(x):
    d, D, N = x[0], x[1], x[2]
    f  = (N + 2) * D * d**2
    g1 = 1 - (D**3 * N) / (71785 * d**4)
    g2 = (4*D**2 - d*D) / (12566*(D*d**3 - d**4)) + 1/(5108*d**2) - 1
    g3 = 1 - 140.45*d / (D**2 * N)
    g4 = (D + d) / 1.5 - 1
    return f + _penalty([g1, g2, g3, g4])

spring = ("Spring_Design", spring_func,
          np.array([0.05, 0.25, 2.0]),
          np.array([2.00, 1.30, 15.0]), 3)


# ── 2. Welded Beam Design ────────────────────────────────────────────────────────

def welded_func(x):
    h, l, t, b = x
    P, L, E, G = 6000, 14, 30e6, 12e6
    tau_m = 6000 / (np.sqrt(2)*h*l)
    M     = P*(L + l/2)
    R     = np.sqrt((l/2)**2 + ((h+t)/2)**2)
    J     = 2*(np.sqrt(2)*h*l*(l**2/12 + ((h+t)/2)**2))
    tau_p = M*R/J
    tau   = np.sqrt(tau_m**2 + 2*tau_m*tau_p*(l/(2*R)) + tau_p**2)
    sigma = 6*P*L/(t**2*b)
    delta = 6*P*L**3/(E*t**3*b)
    Pc    = (4.013*E*np.sqrt(t**2*b**6/36)/L**2)*(1 - t/(2*L)*np.sqrt(E/(4*G)))
    f     = 1.10471*h**2*l + 0.04811*t*b*(14+l)
    g1 = tau - 13600
    g2 = sigma - 30000
    g3 = h - b
    g4 = 0.10471*h**2 + 0.04811*t*b*(14+l) - 5
    g5 = 0.125 - h
    g6 = delta - 0.25
    g7 = P - Pc
    return f + _penalty([g1, g2, g3, g4, g5, g6, g7])

welded = ("Welded_Beam", welded_func,
          np.array([0.1, 0.1, 0.1, 0.1]),
          np.array([2.0, 10.0, 10.0, 2.0]), 4)


# ── 3. Pressure Vessel Design ────────────────────────────────────────────────────

def pressure_vessel_func(x):
    Ts, Th, R, L = x
    Ts = np.round(Ts / 0.0625) * 0.0625
    Th = np.round(Th / 0.0625) * 0.0625
    f  = 0.6224*Ts*R*L + 1.7781*Th*R**2 + 3.1661*Ts**2*L + 19.84*Ts**2*R
    g1 = -Ts + 0.0193*R
    g2 = -Th + 0.00954*R
    g3 = -np.pi*R**2*L - (4/3)*np.pi*R**3 + 1296000
    g4 = L - 240
    return f + _penalty([g1, g2, g3, g4])

pressure = ("Pressure_Vessel", pressure_vessel_func,
            np.array([0.0625, 0.0625, 10.0,  10.0]),
            np.array([6.1875, 6.1875, 200.0, 200.0]), 4)


# ── 4. Speed Reducer Design ──────────────────────────────────────────────────────

def speed_reducer_func(x):
    b, m, z, l1, l2, d1, d2 = x
    f  = (0.7854*b*m**2*(3.3333*z**2 + 14.9334*z - 43.0934)
          - 1.508*b*(d1**2 + d2**2)
          + 7.477*(d1**3 + d2**3)
          + 0.7854*(l1*d1**2 + l2*d2**2))
    g1  = 27/(b*m**2*z) - 1
    g2  = 397.5/(b*m**2*z**2) - 1
    g3  = 1.93*l1**3/(m*z*d1**4) - 1
    g4  = 1.93*l2**3/(m*z*d2**4) - 1
    g5  = np.sqrt((745*l1/(m*z))**2 + 1.69e7)/(0.1*d1**3) - 1100
    g6  = np.sqrt((745*l2/(m*z))**2 + 1.575e8)/(0.1*d2**3) - 850
    g7  = m*z/40 - 1
    g8  = 5*m/b - 1
    g9  = b/(12*m) - 1
    g10 = 1.5*d1 + 1.9 - l1
    g11 = 1.1*d2 + 1.9 - l2
    return f + _penalty([g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11])

speed = ("Speed_Reducer", speed_reducer_func,
         np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0]),
         np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]), 7)


# ── 5. Three-Bar Truss ───────────────────────────────────────────────────────────

def truss_func(x):
    A1, A2 = x
    P, l, sig = 2, 100, 2
    sq2 = np.sqrt(2)
    f   = (2*sq2*A1 + A2) * l
    g1  = (sq2*A1 + A2) / (sq2*A1**2 + 2*A1*A2) * P - sig
    g2  = A2 / (sq2*A1**2 + 2*A1*A2) * P - sig
    g3  = 1/(sq2*A2 + A1) * P - sig
    return f + _penalty([g1, g2, g3])

truss = ("Three_Bar_Truss", truss_func,
         np.array([0.001, 0.001]),
         np.array([1.0,   1.0]), 2)


# ── registry ─────────────────────────────────────────────────────────────────────

ENGINEERING_PROBLEMS = [spring, welded, pressure, speed, truss]
