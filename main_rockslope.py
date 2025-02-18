import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rockslope_example.uspace_rockslope import SubSim
from rockslope_example.ospace_rockslope import frank_tau_to_theta, plackett_tau_to_theta

initial_time = time.time()

# # # %% ---------------------------Some previous steps----------------------------
# # The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

# The following lines are needed to save the final figure in the correc format
# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

# %% Main body of the code
ktau_cphi = -0.75
ktau_zr = -0.5
theta_cphi = frank_tau_to_theta(ktau_cphi)
theta_zr = plackett_tau_to_theta(ktau_zr)

n = 10
n_sim = 5
AnchorTension = np.linspace(0, 700, n)
pf_upp = np.zeros(n)
pf_low = np.zeros(n)

for i in range(n):
    Pfu_temp = np.zeros(n_sim)
    Pfl_temp = np.zeros(n_sim)
    for j in range(n_sim):
        Pfu_temp[j] = SubSim(d=6, b=0, N=100, p0=0.1, bound='upper',
                             g_failure='<=0', Tfrank=theta_cphi,
                             Tplackett=theta_zr, T=AnchorTension[i])
        Pfl_temp[j] = SubSim(d=6, b=0, N=100, p0=0.1, bound='lower',
                             g_failure='<=0', Tfrank=theta_cphi,
                             Tplackett=theta_zr, T=AnchorTension[i])
    pf_upp[i] = np.mean(Pfu_temp)
    pf_low[i] = np.mean(Pfl_temp)
    print(i)

final_time = time.time()

print(f'Elapsed time: {final_time - initial_time}')
print(f'Tension: {AnchorTension}')
print(f'Upper Pf: {pf_upp}')
print(f'Lower Pf: {pf_low}')

cm = 1/2.54
fig = plt.figure(figsize=(15*cm, 15*cm))
plt.plot(AnchorTension, pf_upp, color='C2', label=r'$\overline{P_f}$')
plt.plot(AnchorTension, pf_low, color='C0', label=r'$\underline{P_f}$')
plt.xlabel(r'$T$ (kN)')
plt.ylabel(r'Probability of failure')
plt.xticks(np.linspace(0, 700, 15))
plt.yticks(np.linspace(0, 1, 11))
plt.xlim(0, 700)
plt.ylim(0.000001, 1)
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.show()

# %% Change the directory and save the figure
# os.chdir("./figures/")
# fig.savefig('rockslope_Pfbounds.pgf')
