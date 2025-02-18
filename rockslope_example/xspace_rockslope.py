import numpy as np
import scipy.stats as st
import rockslope_example.gammaBounds as gammaBounds
import rockslope_example.cohesionBounds as cohesionBounds
import rockslope_example.phiBounds as phiBounds
import rockslope_example.zBounds as zBounds
import rockslope_example.rBounds as rBounds
import rockslope_example.khBounds as khBounds


def rock_slope(c, phi, z, r, g, kh, T=0, psi_t=0):
    # c = c
    # print(c, phi, z, r, kh, T, psi_t)
    phi = np.radians(phi)
    # print(c, np.degrees(phi), z, r, kh, T, psi_t)
    H = 20
    psi_f = np.radians(60)
    psi_p = np.radians(35)
    beta = np.radians(np.degrees(psi_p) + psi_t)  # ojo acá, ese 55 es el resultado de 90 - psi_p.
    gamma_t = g # kN/m3
    gamma_w = 9.81  # kN/m3

    A = (H - z)/np.sin(psi_p)
    W = 0.5*gamma_t*(H**2)*((1 - (z/H)**2)*(1/np.tan(psi_p)) - (1/np.tan(psi_f)))
    U = 0.5*gamma_w*r*z*A
    V = 0.5*gamma_w*(r**2)*(z**2)
    N = W*(np.cos(psi_p) - kh*np.sin(psi_p)) - U - V*np.sin(psi_p) + T*np.sin(beta)

    Fs = (c*A + N*np.tan(phi))/(W*(np.sin(psi_p) + kh*np.cos(psi_p)) + V*np.cos(psi_p) - T*np.cos(beta))
    # print(Fs)
    # print(Fs)
    return Fs - 1


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
        z0 = focal_elements[i, 4]
        z1 = focal_elements[i, 5]
        r = focal_elements[i, 6]
        g0 = focal_elements[i, 7]
        g1 = focal_elements[i, 8]
        kh = focal_elements[i, 9]

        vector_c = st.uniform.rvs(size=n_mcs, loc=c0, scale=c1-c0)
        vector_phi = st.uniform.rvs(size=n_mcs, loc=phi0, scale=phi1-phi0)
        vector_z = st.uniform.rvs(size=n_mcs, loc=z0, scale=z1-z0)
        vector_g = st.uniform.rvs(size=n_mcs, loc=g0, scale=g1-g0)

        for j in range(n_mcs):
            g_mcs[j] = rock_slope(c=vector_c[j], phi=vector_phi[j],
                                  z=vector_z[j], r=r, g=vector_g[j], kh=kh, T=T, psi_t=0)

        g[i] = critical_value(g_mcs)

    return g


def transformation_omega_to_x(omega_vector):
    N = omega_vector.shape[0]

    c_FocElems = cohesionBounds.c_FocElems(omega_vector[:,0])
    phi_FocElems = phiBounds.phi_FocElems(omega_vector[:,1])
    z_FocElems = zBounds.z_FocElems(omega_vector[:,2])
    r_FocElems = rBounds.r_FocElems(omega_vector[:,3])
    gamma_FocElems = gammaBounds.g_FocElems(omega_vector[:,4])
    kh_FocElems = khBounds.kh_FocElems(omega_vector[:,5])

    foc_elmnts = np.concatenate((c_FocElems, phi_FocElems, z_FocElems, 
                                 r_FocElems, gamma_FocElems, kh_FocElems), axis=1)

    # focal_elements[i, :] = (c0, c1, phi0, phi1, z0, z1, r, g0, g1 kh)

    return foc_elmnts
