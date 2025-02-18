import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from foundation_example.uspace_foundation import SubSim
from foundation_example.ospace_foundation import frank_tau_to_theta, plackett_tau_to_theta

initial_time = time.time()

# # # %% ---------------------------Some previous steps----------------------------
# # The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

# The following lines are needed to save the final figure in the correc format
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# %% Main body of the code
ktau_cphi = -0.75
theta_cphi = plackett_tau_to_theta(ktau_cphi)

n = 10
n_sim = 15
foundationWidth = np.linspace(1, 2.5, n)
pf_upp = np.zeros(n)
pf_low = np.zeros(n)

for i in range(n):
    Pfu_temp = np.zeros(n_sim)
    Pfl_temp = np.zeros(n_sim)
    for j in range(n_sim):
        Pfu_temp[j] = SubSim(d=3, b=0, N=100, p0=0.1, bound='upper',
                             g_failure='<=0', Tplackett=theta_cphi, 
                             T=foundationWidth[i])
        Pfl_temp[j] = SubSim(d=3, b=0, N=100, p0=0.1, bound='lower',
                             g_failure='<=0', Tplackett=theta_cphi, 
                             T=foundationWidth[i])
    pf_upp[i] = np.mean(Pfu_temp)
    pf_low[i] = np.mean(Pfl_temp)
    print(pf_upp[i],pf_low[i])
    print(i)

final_time = time.time()

print(f'Elapsed time: {final_time - initial_time}')
print(f'foundationWidth: {foundationWidth}')
print(f'Upper Pf: {pf_upp}')
print(f'Lower Pf: {pf_low}')

cm = 1/2.54
fig = plt.figure(figsize=(15*cm, 15*cm))
plt.plot(foundationWidth, pf_upp, color='C2', label=r'$\overline{P_f}$')
plt.plot(foundationWidth, pf_low, color='C0', label=r'$\underline{P_f}$')
plt.xlabel(r'Footing width (B)')
plt.ylabel(r'Probability of failure')
plt.xticks(np.linspace(1, 2.5, 7))
plt.yticks(np.linspace(0, 1, 11))
plt.xlim(1, 2.5)
plt.ylim(0.001, 1)
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle=':')
# plt.show()

# %% Change the directory and save the figure
os.chdir("./figures/")
fig.savefig('foundation_Pfbounds.pdf')
