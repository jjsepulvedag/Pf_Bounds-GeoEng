import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cslope_example.uspace_cslope import SubSim
from cslope_example.ospace_cslope import transformation_u_to_omega, tau_to_theta
from cslope_example.xspace_cslope import transformation_omega_to_x, phi_critical

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

# %% Computing upper and lower Pf and obtain the u-vector of each intermediate 
# level of the SubSim 

ktau = +0.1  # Kendall's tau coefficient between cohesion and friction angle
tn16 = tau_to_theta(ktau)
Num_Sim = 1
pf_low = np.zeros(Num_Sim)
pf_upp = np.zeros(Num_Sim)
for i in range(Num_Sim):
    pf_low[i], u_samples_low, glow = SubSim(d=2, b=0, N=1000, p0=0.1,
                                            bound='lower', g_failure='<=0',
                                            tn16=tn16)
    pf_upp[i], u_samples_upp, gupp = SubSim(d=2, b=0, N=1000, p0=0.1,
                                            bound='upper', g_failure='<=0',
                                            tn16=tn16)
    print(i)

# print(u_samples_low.shape)
# print(u_samples_upp.shape)
# print(u_samples_low)
               
print(f'Upper probability of failure equal to: {pf_upp}')
print(f'Lower probability of failure equal to: {pf_low}')
print(f'Mean upper Pf: {np.mean(pf_upp)}')
print(f'Mean lower Pf: {np.mean(pf_low)}')

# %% From the u-vector (standard gaussian space) obtain the o-vector (omega 
# space) and the focal elements (x-space)

o_samples_low = np.zeros(u_samples_low.shape)
for i in range(o_samples_low.shape[2]):
    o_samples_low[:, :, i] = transformation_u_to_omega(u_samples_low[:, :, i],
                                                       theta=tn16)

x_samples_low = np.zeros((u_samples_low.shape[0], 4, u_samples_low.shape[2]))
for i in range(x_samples_low.shape[2]):
    x_samples_low[:, :, i] = transformation_omega_to_x(o_samples_low[:, :, i])

o_samples_upp = np.zeros(u_samples_upp.shape)
for i in range(o_samples_upp.shape[2]):
    o_samples_upp[:, :, i] = transformation_u_to_omega(u_samples_upp[:, :, i],
                                                       theta=tn16)

x_samples_upp = np.zeros((u_samples_upp.shape[0], 4, u_samples_upp.shape[2]))
for i in range(x_samples_upp.shape[2]):
    x_samples_upp[:, :, i] = transformation_omega_to_x(o_samples_upp[:, :, i])





# %% Plotting area

cm = 1/2.54
fig1 = plt.figure(figsize=(16.5*cm, 18*cm))
# fig2 = plt.figure(figsize=(16.5*cm, 6*cm))
# fig3 = plt.figure(figsize=(16.5*cm, 6*cm))
ax1 = fig1.add_subplot(3, 2, 1)
ax2 = fig1.add_subplot(3, 2, 2)
ax3 = fig1.add_subplot(3, 2, 3)
ax4 = fig1.add_subplot(3, 2, 4)
ax5 = fig1.add_subplot(3, 2, 5)
ax6 = fig1.add_subplot(3, 2, 6)

# colors = ['C0', 'C8', 'C1', 'C3']
colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red']
labels = [r'SS level $0$', r'SS level $1$', r'SS level $2$', r'SS level $3$']

for i in range(u_samples_low.shape[2]):
    ax1.scatter(u_samples_low[:, 0, i], u_samples_low[:, 1, i], color=colors[i], s=3, label=labels[i])
ax1.set_xticks(np.linspace(-4, 4, 9))
ax1.set_yticks(np.linspace(-6, 6, 7))
ax1.set_xlim((-4, 4))
ax1.set_ylim((-6, 6))
ax1.set_xlabel(r'$u_1$')
ax1.set_ylabel(r'$u_2$')
ax1.set_title(r'$\mathscr{U}$-space')

for i in range(u_samples_upp.shape[2]):
    ax2.scatter(u_samples_upp[:, 0, i], u_samples_upp[:, 1, i], color=colors[i], s=3, label=labels[i])
ax2.set_xticks(np.linspace(-4, 4, 9))
ax2.set_yticks(np.linspace(-6, 6, 7))
ax2.set_xlim((-4, 4))
ax2.set_ylim((-6, 6))
ax2.set_xlabel(r'$u_1$')
ax2.set_ylabel(r'$u_2$')
ax2.set_title(r'$\mathscr{U}$-space')

for i in range(o_samples_low.shape[2]):
    ax3.scatter(o_samples_low[:, 0, i], o_samples_low[:, 1, i], color=colors[i], s=3, label=labels[i])
ax3.set_xticks(np.linspace(0, 1, 6))
ax3.set_yticks(np.linspace(0, 1, 6))
ax3.set_xlim((0, 1))
ax3.set_ylim((0, 1))
ax3.set_xlabel(r'$\alpha_1$')
ax3.set_ylabel(r'$\alpha_2$')
ax3.set_title(r'$\Omega$-space')

for i in range(o_samples_upp.shape[2]):
    ax4.scatter(o_samples_upp[:, 0, i], o_samples_upp[:, 1, i], color=colors[i], s=3, label=labels[i])
ax4.set_xticks(np.linspace(0, 1, 6))
ax4.set_yticks(np.linspace(0, 1, 6))
ax4.set_xlim((0, 1))
ax4.set_ylim((0, 1))
ax4.set_xlabel(r'$\alpha_1$')
ax4.set_ylabel(r'$\alpha_2$')
ax4.set_title(r'$\Omega$-space')

for i in range(x_samples_low.shape[2]):
    for k in range(x_samples_low.shape[0]):
        x0 = x_samples_low[k, 0, i]
        y0 = x_samples_low[k, 2, i]
        w = x_samples_low[k, 1, i] - x0
        h = x_samples_low[k, 3, i] - y0
        rect = mpl.patches.Rectangle((x0, y0), width=w, height=h, linewidth=0.5,
                                     edgecolor=colors[i], facecolor='none',
                                     linestyle='dashed')
        ax5.add_patch(rect)
    rect = mpl.patches.Rectangle((x0, y0), width=w, height=h, linewidth=0.5,
                                 edgecolor=colors[i], facecolor='none',
                                 linestyle='dashed', label=labels[i])
    ax5.add_patch(rect)

ax5.set_xticks(np.linspace(5, 35, 7))
ax5.set_yticks(np.linspace(0, 45, 4))
ax5.set_xlim((5, 35))
ax5.set_ylim((0, 45))
ax5.set_xlabel(r'$\gamma \left(\operatorname{kN/m}^3\right)$')
ax5.set_ylabel(r'$\phi\left(^{\circ}\right)$')
ax5.set_title(r'$\mathscr{X}$-space')

for i in range(x_samples_upp.shape[2]):
    for k in range(x_samples_upp.shape[0]):
        x0 = x_samples_upp[k, 0, i]
        y0 = x_samples_upp[k, 2, i]
        w = x_samples_upp[k, 1, i] - x0
        h = x_samples_upp[k, 3, i] - y0
        rect = mpl.patches.Rectangle((x0, y0), width=w, height=h, linewidth=0.5,
                                     edgecolor=colors[i], facecolor='none',
                                     linestyle='dashed')
        ax6.add_patch(rect)
    rect = mpl.patches.Rectangle((x0, y0), width=w, height=h, linewidth=0.5,
                                 edgecolor=colors[i], facecolor='none',
                                 linestyle='dashed', label=labels[i])
    ax6.add_patch(rect)

ax6.set_xticks(np.linspace(5, 35, 7))
ax6.set_yticks(np.linspace(0, 45, 4))
ax6.set_xlim((5, 35))
ax6.set_ylim((0, 45))
ax6.set_xlabel(r'$\gamma \left(\operatorname{kN/m}^3\right)$')
ax6.set_ylabel(r'$\phi\left(^{\circ}\right)$')
ax6.set_title(r'$\mathscr{X}$-space')


critical_gamma = np.linspace(7, 35, 100)
critical_phis = np.zeros(len(critical_gamma))
for i in range(len(critical_gamma)):
    critical_phis[i] = phi_critical(critical_gamma[i])
ax5.plot(critical_gamma, critical_phis, color='k', label=r'$G\left(\gamma,\phi\right)$')
ax6.plot(critical_gamma, critical_phis, color='k', label=r'$G\left(\gamma,\phi\right)$')


ax1.legend(loc='upper right', fontsize='x-small', ncol=2)
ax2.legend(loc='upper right', fontsize='x-small')
ax3.legend(loc='upper right', fontsize='x-small', ncol=2)
ax4.legend(loc='upper right', fontsize='x-small')
ax5.legend(loc='upper right', fontsize='x-small', ncol=2)
ax6.legend(loc='upper right', fontsize='x-small')
plt.tight_layout()
plt.show()

# %% Change the directory and save the figure
# os.chdir("../1.0_Figures/")
# fig1.savefig('SuS_evolution.pdf')