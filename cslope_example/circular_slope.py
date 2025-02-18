import numpy as np
import scipy.optimize as opt


def phi_critical(cohesion):
    c = cohesion

    def phi_crit(phi):
        N_slices = 11
        kh = 0.0
        gamma_t = 20  # kN/m3
        gamma_w = 9.81  # kN/m3
        # c = 15  # kPa
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
