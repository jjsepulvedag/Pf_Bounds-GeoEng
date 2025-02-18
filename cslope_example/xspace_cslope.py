import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import cslope_example.phiBounds_KS as phiBounds_KS
import cslope_example.gammaBounds_FZ as gammaBounds_FZ


def lim_state_funct(x1, x2):
    N_slices = 11
    kh = 0
    gamma_t = x1 # kN/m3
    gamma_w = 9.81  # kN/m3
    c = 0  # kPa
    phi = np.radians(x2)  # °

    Areas = np.array([3.08, 7.81, 11.31, 16.50, 21.14, 26.64, 24.5, 21.81,
                      20.38, 17.76, 10])
    # Widths = np.array([29/11]*11)
    Alphas = np.array([-39.65, -28.80, -19.01, -9.80, -0.84, 8.11, 17.26,
                       26.90, 37.5, 50.04, 70.23])
    Alphas = np.radians(Alphas)
    Hws = np.array([1.16, 2.97, 4.14, 4.82, 5.10, 4.90, 4.31, 3.23, 1.56, 0.00,
                    0.00])
    Ls = np.array([3.43, 3.01, 2.79, 2.68, 2.64, 2.67, 2.76, 2.96, 3.33, 4.12,
                   7.86])

    Wts = Areas*gamma_t
    uws = Hws*gamma_w
    Us = uws*Ls

    numerators = np.zeros(11)
    denominators = np.zeros(11)

    for i in range(N_slices):
        numerators[i] = (c*Ls[i] + (Wts[i]*np.cos(Alphas[i]) -
                         kh*Wts[i]*np.sin(Alphas[i]) - Us[i])*np.tan(phi))
        denominators[i] = (Wts[i]*np.sin(Alphas[i]) +
                           kh*Wts[i]*np.cos(Alphas[i]))

    lsf = sum(numerators)/sum(denominators) - 1
    return lsf


def failure_prob(focal_elements, bound='upper'):

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

        x0 = focal_elements[i, 0]
        x1 = focal_elements[i, 1]
        y0 = focal_elements[i, 2]
        y1 = focal_elements[i, 3]
        try:
            vector_x = st.uniform.rvs(size=n_mcs, loc=x0, scale=x1-x0)
            vector_y = st.uniform.rvs(size=n_mcs, loc=y0, scale=y1-y0)
        except ValueError:
            continue

        for j in range(n_mcs):
            g_mcs[j] = lim_state_funct(vector_x[j], vector_y[j])

        g[i] = critical_value(g_mcs)
        # g[i] = critical_value(g_temp)
    # print(g)
    return g


def transformation_omega_to_x(omega_vector):
    N = omega_vector.shape[0]

    gamma_FocElems = gammaBounds_FZ.gamma_FocElems(omega_vector[:,0])
    phi_FocElems = phiBounds_KS.phi_FocElems(omega_vector[:,1])

    # foc_elmnts = np.concatenate((phi_FocElems, gamma_FocElems), axis=1)
    foc_elmnts = np.concatenate((gamma_FocElems, phi_FocElems), axis=1)

    return foc_elmnts


def phi_critical(gamma_t):

    def phi_crit(phi):
        N_slices = 11
        kh = 0.00
        # gamma_t = 20  # kN/m3
        gamma_w = 9.81  # kN/m3
        c = 0  # kPa
        phi = np.radians(phi)  # 20°

        Areas = np.array([3.08, 7.81, 11.31, 16.50, 21.14, 26.64, 24.5, 21.81,
                          20.38, 17.76, 10])
        # Widths = np.array([29/11]*11)
        Alphas = np.array([-39.65, -28.80, -19.01, -9.80, -0.84, 8.11, 17.26,
                           26.90, 37.5, 50.04, 70.23])
        Alphas = np.radians(Alphas)
        Hws = np.array([1.16, 2.97, 4.14, 4.82, 5.10, 4.90, 4.31, 3.23, 1.56,
                        0.00, 0.00])
        Ls = np.array([3.43, 3.01, 2.79, 2.68, 2.64, 2.67, 2.76, 2.96, 3.33,
                       4.12, 7.86])

        Wts = Areas*gamma_t
        uws = Hws*gamma_w
        Us = uws*Ls

        numerators = np.zeros(11)
        denominators = np.zeros(11)

        for i in range(N_slices):
            numerators[i] = (c*Ls[i] + (Wts[i]*np.cos(Alphas[i]) -
                             kh*Wts[i]*np.sin(Alphas[i]) - Us[i])*np.tan(phi))
            denominators[i] = (Wts[i]*np.sin(Alphas[i]) +
                               kh*Wts[i]*np.cos(Alphas[i]))

        fs = sum(numerators)/sum(denominators) - 1
        return fs

    phi_critical = opt.fsolve(phi_crit, 20)

    return phi_critical
