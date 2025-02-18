import numpy as np
import scipy.stats as st
import matplotlib as mpl
import foundation_example.ospace_foundation as Ospace
import foundation_example.xspace_foundation as Xspace
# from ospace_example1 import transformation_u_to_omega
# from xspace_example1 import transformation_omega_to_x, failure_prob


def mma(theta0, g0, N, b, spread=2, bound='upper', g_failure='>=', 
        Tplackett=0.1, T=0):

    if g_failure == '>=0':
        sign_g = 1
    elif g_failure == '<=0':
        sign_g = -1
    else:
        raise Exception("g_failure must be either '>x=0' or '<=0'.")

    d = len(theta0)  # Number of parameters (dimension of theta)
    spreads = d*[spread]
    pim = st.norm(loc=0, scale=1)

    theta = np.zeros((N, d));  theta[0, :] = theta0
    g     = np.zeros(N);       g[0]        = g0

    for i in range(N - 1):
        # Generate a candidate state \xi
        xi = np.zeros((1, d))

        for k in range(d):
            # the proposal PDFs are defined (the must be symmetric)
            # we will use a uniform PDF in [loc, loc+scale]
            Sk = st.uniform(loc=theta[i, k] - spreads[k], scale=2*spreads[k])

            # Draw a sample from the proposal Sk
            hat_xi_k = Sk.rvs()

            # compute the acceptance ratio
            r = pim.pdf(hat_xi_k)/pim.pdf(theta[i, k])  # eq. 8

            # acceptance/rejection step:  # eq. 9
            if np.random.rand() <= min(1, r):
                xi[0, k] = hat_xi_k     # accept the candidate
            else:
                xi[0, k] = theta[i, k]  # reject the candidate

        # check whether xi \in F by system analysis
        alpha_sample = Ospace.transformation_u_to_omega(u_vector=xi, 
                                                        theta_plackett=Tplackett)
        x_sample = Xspace.transformation_omega_to_x(alpha_sample)
        # gg = sign_g*g_lim(*xi, *args)
        gg = sign_g*Xspace.failure_prob(x_sample, T, bound=bound)
        if gg > b:  # eq. 10
            # xi belongs to the failure region
            theta[i+1, :] = xi
            g[i+1] = gg
        else:
            # xi does not belong to the failure region
            theta[i+1, :] = theta[i, :]
            g[i+1] = g[i]

    # return theta and its corresponding g
    return theta, g


def SubSim(d=3, b=0, N=1000, p0=0.1, bound='upper', g_failure='>=0', 
           Tplackett=0.1, T=0):
    
    '''
    d: dimension
    '''

    if g_failure == '>=0':
        sign_g = 1
    elif g_failure == '<=0':
        sign_g = -1
    else:
        raise Exception("g_failure must be either '>=0' or '<=0'.")

    Nc = int(N*p0)  # number of Markov chains (number of seeds per level)
    Ns = int(1/p0)  # number of samples per Markov chain, including the seed

    j = 0  # Number of conditional levels, j = 0 crude MCS leven F_0
    N_f = np.array([])  # Number of failure samples at level j
    theta = np.zeros((N, d, 1))  # List of samples u at each level j
    g = np.zeros((N, 1))  # Evaluation of each seed of samples at each level j
    b_j = np.array([])  # Intermediate threshold values

    # Create the d-dimensional standard Gaussian distribution (sgd)
    mean_sgd = np.zeros(d)  # mean of the d-dimensional stan Gauss Dist (sgd)
    cov_sgd = np.eye(d)  # cov of the d-dimensional stan Gaussian dist (sgd)
    sgd = st.multivariate_normal(mean=mean_sgd, cov=cov_sgd)

    # Draw N i.i.d. samples from pi = pi(.|F0) using MCS
    theta[:, :, 0] = sgd.rvs(N)

    alpha_samples = Ospace.transformation_u_to_omega(u_vector=theta[:, :, j],
                                                     theta_plackett=Tplackett)
    x_samples = Xspace.transformation_omega_to_x(alpha_samples)

    g[:, j] = sign_g*Xspace.failure_prob(x_samples, T=T, bound=bound)

    # Count the number of samples in level F[0] and append it to N_f
    N_f = np.append(N_f, np.sum(g[:, j] > sign_g*b))  # b = 0
    # print(N_f[j])
    while N_f[j]/N < p0:

        # sort the limit state values in ascending order
        idx = np.argsort(g[:, j])  # index of the sorting
        g_sorted = g[:, j][idx]    # sort g_j using the idx key
        theta_sorted = theta[:, :, j][idx, :]  # sort theta_j using idx key

        # estimate the p0-percentile of g and set it as the b_j
        b_j = np.append(b_j, None)
        b_j[j] = (g_sorted[N-Nc] + g_sorted[N-Nc+1])/2

        # select the seeds: they are the last Nc samples associated to idx
        tseed = theta_sorted[-Nc:, :]  # seed of theta samples
        gseed = g_sorted[-Nc:]  # corresponding g of theta samples

        # starting from seed[k,:] draw Ns-1 additional samples from pi(.|Fj)
        # using a MCMC algorithm called MMA
        theta_from_seed = np.zeros((Ns, d, Nc))
        g_from_seed     = np.zeros((Ns, Nc))  # Nc*[None]

        for k in range(Nc):  # Nc = N*p0 = number of chains
            theta_from_seed[:, :, k], g_from_seed[:, k] = \
                mma(tseed[k, :], gseed[k], Ns, b_j[j], spread=2, bound=bound,
                    g_failure=g_failure, Tplackett=Tplackett, T=T)

        # concatenate all samples theta_from_seed[k] in a single array theta
        theta = np.append(theta, np.zeros((N, d, 1)), axis=2)
        theta_from_seed = theta_from_seed.transpose(2, 0, 1).reshape(N, d)
        theta[:, :, j + 1] = theta_from_seed

        g = np.append(g, np.zeros((N, 1)), axis=1)
        g_from_seed = g_from_seed.transpose(1, 0).reshape(N)
        g[:, j + 1] = g_from_seed

        # Count the number of samples in level F[j+1] and append it to N_f
        N_f = np.append(N_f, np.sum(g[:, j + 1] > sign_g*b))  # b = 0
        # continue with the next intermediate failure level
        j += 1
        # print(N_f[j])

    pf = (p0**j) * (N_f[j]/N)
    return pf
