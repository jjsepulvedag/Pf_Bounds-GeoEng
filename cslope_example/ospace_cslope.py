import numpy as np
import scipy.stats as st
import scipy.optimize as sop

def rvs(ind_vector, theta):
    r"""Random sampling from the No16 copula

    This function draws random samples that follow the dependence
    structure of a No16 copula (Nelsen, 2006).

    Parameters:
    ----------
    theta : float
        Copula dependence parameter in [0,inf)
    n : int
        Numbers of samples to be generated
    Returns:
    ----------
    u : numpy.ndarray
        vector of n elements on [0,1], which with v follows a No16 cop
    v : numpy.ndarray
        vector of n elements on [0,1], which with u follows a No16 cop
    Notes:
    ----------
    """
    
    def bisection(k):
        def function(t):
            return k - (2*theta/t - theta + 1)/(theta/(t**2)+1)
        solution = sop.bisect(function, 1e-16, 1-1e-16)
        return solution

    theta = np.round(theta, 3)

    s = ind_vector[:, 0]
    q = ind_vector[:, 1]

    t = np.fromiter(map(bisection, q), dtype=np.float64)

    generator = (theta/t + 1)*(1 - t)

    t_u = s*generator
    t_v = (1 - s)*generator

    u = 1/2 - t_u/2 - theta/2 + np.sqrt((t_u + theta - 1)**2 + 4*theta)/2
    v = 1/2 - t_v/2 - theta/2 + np.sqrt((t_v + theta - 1)**2 + 4*theta)/2

    return u, v


def transformation_u_to_omega(u_vector, theta):

    standard_normal = st.norm(loc=0, scale=1)
    ind_vector = standard_normal.cdf(u_vector)
    dep_vector = np.zeros((u_vector.shape[0], u_vector.shape[1]))

    dep_vector[:, 0], dep_vector[:, 1] = rvs(ind_vector, theta)

    return dep_vector


def tau_to_theta(tau):
    def function(t):
        f11 = 2*np.arctan(2*np.sqrt(t/4)*(t-1)/(t-t**2))
        f12 = np.sqrt(t/4)*(t-1)
        f1 = f11*f12
        value = tau-1+4*(t*np.log(t+1)-(t-1)-0.5-f1-t*np.log(t))
        return value

    solution = sop.bisect(function, a=1e-10, b=1e10)
    return solution




# gaussian_dist = st.norm(loc=0, scale=1)

# u_vector = gaussian_dist.rvs(size=(10000, 2))
# plt.scatter(u_vector[:, 0], u_vector[:, 1])
# plt.show()

# o_vector = transformation_u_to_omega(u_vector=u_vector, theta=0.014)

# print(st.kendalltau(o_vector[:, 0], o_vector[:, 1]))


# plt.scatter(o_vector[:, 0], o_vector[:, 1])
# plt.show()