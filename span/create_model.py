from span import rotin_inp as ri
import numpy as np

starA=True
starB=False

if starA==True:
    teffA = np.linspace(10, 15, 6, dtype=int)
    # rotA = np.linspace(10, 70, 7, dtype=int)
    rotA = list(range(0, 650, 50))
    for T1 in teffA:
        if T1 < 16:
            loggA = [18, 20, 22, 24, 26, 28, 30]
            for g1 in loggA:
                for r1 in rotA:
                    ri.rotin3_inp('lateB', T1, g1, r1)
        else:
            loggA = [20, 22.5, 25, 27.5, 30]
            for g1 in loggA:
                for r1 in rotA:
                    ri.rotin3_inp('earlyB', T1, g1, r1)

if starB==True:
    teffB = [16, 18, 20, 22, 24, 26, 28, 30, 32.5, 35, 37.5]
    # teffB = [16, 20]
    loggB = [20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45]
    # loggB = [42.5, 45]
    # rotB = range(0, 600, 50)
    rotB = [600]
    for T2 in teffB:
        for g2 in loggB:
            for r2 in rotB:
                if T2>30 and g2!=30:
                    ri.rotin3_inp('O', T2, g2, r2)
                else:
                    ri.rotin3_inp('earlyB', T2, g2, r2)