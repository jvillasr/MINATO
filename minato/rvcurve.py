import numpy as np
import kepler

def rvcurve(t, P, Tp, e, omega, gamma, K1, K2, SB2=False):
    omega = (omega)*np.pi/180
    # print('omega =', omega)

    '''Mean anomaly'''
    M = (2*np.pi/P) * (t-Tp)
    for i in range(len(M)):
        while M[i] < 0:
            M[i] = M[i] + 2*np.pi
        while M[i] > 2*np.pi:
            M[i] = M[i] - 2*np.pi
    # print(M)
    M_scl = ( ((M - min(M))/(max(M)-min(M))) * 2*np.pi )
    # print(M_scl)

    '''Eccentric anomaly'''
    E = kepler.solve(M_scl, e)
    # E = kepler.solve(M, e)
    # print(E)
    for x in E:
        if x <= 0 or x>2*np.pi:
            print(x)

    # from scipy.optimize import fsolve
    # def func(E, *pars):
    #     e, M = pars
    #     return E - e*np.sin(E) - M
    # # params = (e, M)
    # E = [fsolve(func,100, args=(e, m)) for m in M_scl]
    # E = np.array([x[0] for x in E])
    # print(E)

    '''True anomaly'''
    # theta = [2 * np.arctan( np.sqrt( (1+e)/(1-e) ) * np.tan(x/2) ) for x in E]
    theta = 2 * np.arctan( np.sqrt( (1+e)/(1-e) ) * np.tan(E/2) )
    # print(theta)

    '''RVs'''
    v1 = gamma + K1 * ( np.cos(theta+omega) + e*np.cos(omega) )
    # print(np.cos(theta+omega))
    # print(v1)
    # if SB2==True:
    omega2 = omega + np.pi
    v2 = gamma + K2 * ( np.cos(theta+omega2) + e*np.cos(omega2) )

    return v1, v2
    # else:
    #     return v1
