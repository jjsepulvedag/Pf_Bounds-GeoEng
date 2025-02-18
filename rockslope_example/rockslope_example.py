import numpy as np

def limit_state_funct(cohesion, phi, z, r, kh=0, T=0, psi_t=0):
    c = cohesion
    phi = np.radians(phi)
    H = 12
    psi_f = np.radians(60)
    psi_p = np.radians(35)
    beta = np.radians(35 + psi_t)  # ojo acá, ese 55 es el resultado de 90 - psi_p.
    gamma_t = 26  # kN/m3
    gamma_w = 9.81  # kN/m3

    A = (H - z)/np.sin(psi_p)
    W = 0.5*gamma_t*(H**2)*((1 - (z/H)**2)*(1/np.tan(psi_p)) - (1/np.tan(psi_f)))
    print(W, )
    U = 0.5*gamma_w*r*z*A
    V = 0.5*gamma_w*(r**2)*(z**2)
    N = W*(np.cos(psi_p) - kh*np.sin(psi_p)) - U - V*np.sin(psi_p) + T*np.sin(beta)
    print(np.degrees(beta))

    Fs = (c*A + N*np.tan(phi))/(W*(np.sin(psi_p) + kh*np.cos(psi_p)) + V*np.cos(psi_p) - T*np.cos(beta))
    print(phi)
    print((c*A + N*np.tan(phi)))
    return Fs

print(limit_state_funct(cohesion=0, phi=37, z=4.35, r=0, kh=0.0, T=400, psi_t=2))
