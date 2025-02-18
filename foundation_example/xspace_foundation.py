import numpy as np
import scipy.stats as st
import foundation_example.pBounds as pBounds
import foundation_example.cohesionBounds as cohesionBounds
import foundation_example.phiBounds as phiBounds

def foundation(c, phi, P, B):
    # Terzaghi square footing equation, see Bowles (1997) table 4-1
    # c: Cohesion, phi: friction angle, g: unit weight

    df = 2
    g = 20 # kn/m3
    phi = np.radians(phi)

    a = np.exp((3*np.pi/4 - phi/2)*np.tan(phi))
    # print(a)
    Nq = (a**2)/(2*np.cos(np.pi/4 + phi/2)**2)
    # print(Nq)

    Nc = (Nq - 1)/np.tan(phi)
    # print(Nc)

    kp = np.tan(np.pi/4 + phi/2)**2
    Ng = (np.tan(phi)/2)*(kp/np.cos(phi)**2 - 1)
    # print(Ng)
    # Ng = 8.34

    qu = 1.3*c*Nc + g*df*Nq + 0.4*g*B*Ng

    limit_state = qu - P/B**2

    return limit_state



# def rock_slope(c, phi, z, r, g, kh, T=0, psi_t=0):
#     # c = c
#     # print(c, phi, z, r, kh, T, psi_t)
#     phi = np.radians(phi)
#     # print(c, np.degrees(phi), z, r, kh, T, psi_t)
#     H = 20
#     psi_f = np.radians(60)
#     psi_p = np.radians(35)
#     beta = np.radians(np.degrees(psi_p) + psi_t)  # ojo acá, ese 55 es el resultado de 90 - psi_p.
#     gamma_t = g # kN/m3
#     gamma_w = 9.81  # kN/m3

#     A = (H - z)/np.sin(psi_p)
#     W = 0.5*gamma_t*(H**2)*((1 - (z/H)**2)*(1/np.tan(psi_p)) - (1/np.tan(psi_f)))
#     U = 0.5*gamma_w*r*z*A
#     V = 0.5*gamma_w*(r**2)*(z**2)
#     N = W*(np.cos(psi_p) - kh*np.sin(psi_p)) - U - V*np.sin(psi_p) + T*np.sin(beta)

#     Fs = (c*A + N*np.tan(phi))/(W*(np.sin(psi_p) + kh*np.cos(psi_p)) + V*np.cos(psi_p) - T*np.cos(beta))
#     # print(Fs)
#     # print(Fs)
#     return Fs - 1


def failure_prob(focal_elements, T, bound='upper'):

    if bound == 'upper':
        critical_value = np.min
    elif bound == 'lower':
        critical_value = np.max
    else:
        print('Invalid bound. Please choose \"upper\" or \"lower\"')

    N = focal_elements.shape[0]
    n_mcs = 100  # -----------------------------------------------------------
    g_mcs = np.zeros(n_mcs)  # ------------------------------------------------
    g = np.zeros(N)
    # g_temp = np.zeros(4)

    for i in range(N):
        # print(focal_elements[i, :])
        c0 = focal_elements[i, 0]
        c1 = focal_elements[i, 1]
        phi0 = focal_elements[i, 2]
        phi1 = focal_elements[i, 3]
        p = focal_elements[i, 4]

        vector_c = st.uniform.rvs(size=n_mcs, loc=c0, scale=c1-c0)
        vector_phi = st.uniform.rvs(size=n_mcs, loc=phi0, scale=phi1-phi0)


        for j in range(n_mcs):
            g_mcs[j] = foundation(c=vector_c[j], phi=vector_phi[j],
                                 P=p, B=T)

        g[i] = critical_value(g_mcs)

    return g


def transformation_omega_to_x(omega_vector):
    N = omega_vector.shape[0]

    c_FocElems = cohesionBounds.c_FocElems(omega_vector[:,0])
    phi_FocElems = phiBounds.phi_FocElems(omega_vector[:,1])
    p_FocElems = pBounds.p_FocElems(omega_vector[:,2])

    foc_elmnts = np.concatenate((c_FocElems, phi_FocElems, p_FocElems), axis=1)


    return foc_elmnts
