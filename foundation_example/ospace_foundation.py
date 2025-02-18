import numpy as np
import scipy.stats as st
import scipy.optimize as sop
import scipy.integrate as sint

def rvs_frank(ind_vector, theta):
    r"""Random sampling from the Frank Copula.

    This function draws random samples that follow the dependence
    structure of a Frank copula.

    Parameters:
    ----------
    theta : float
        Copula dependence parameter in (-inf,inf)
    n : int
        Numbers of samples to be generated
    Returns:
    ----------
    u : numpy.ndarray
        vector of n elements on [0,1], which with v follows a Frank cop
    v : numpy.ndarray
        vector of n elements on [0,1], which with u follows a Frank cop
    Notes:
    ----------
    """

    u = ind_vector[:, 0]
    w = ind_vector[:, 1]

    v = (-(1/theta) * np.log(1 + w*(1-np.exp(-theta))
         / (w*(np.exp(-theta*u)-1)-np.exp(-theta*u))))

    return u, v


def rvs_plackett(ind_vector, theta):
    """Random sampling from Plackett copula.

    This function draws random samples that follow the dependence
    structure of a Plackett copula. It is restricted to a bivariate
    dimension uniquely.

    Parameters:
    ----------
    theta : float
        Parameter of dependence of the bivariate Plackett copula
    n : int
        Numbers of samples to be generated
    Returns:
    ----------
    u : numpy.ndarray
        vector of n elements on [0,1], which with v follow a Plackett cop
    v : numpy.ndarray
        vector of n elements on [0,1], which with u follow a Plackett cop
    Notes:
    ----------
    """

    u = ind_vector[:, 0]
    t = ind_vector[:, 1]

    a = t*(1 - t)
    b = theta + a*(theta - 1)**2
    c = 2*a*(u*theta**2 + 1 - u) + theta*(1 - 2*a)
    d = np.sqrt(theta)*np.sqrt(theta + 4*a*u*(1 - u)*(1 - theta)**2)

    v = (c - (1 - 2*t)*d)/(2*b)

    return u, v


def frank_tau_to_theta(tau):

    def bisection(tau):
        def function(theta):
            def debye(t):
                value = t/(np.exp(t) - 1)
                return value
            integration = sint.quad(debye, a=0, b=theta)[0]/theta
            return tau - (1 - (4/theta)*(1-integration))

        if -1+1e-3 < tau < 1-1e-3:
            sol = sop.bisect(function, -500, 501)
            return sol
        elif tau <= -1+1e-3:
            return -np.inf
        elif 1-1e-3 <= tau:
            return np.inf

    theta = bisection(tau)

    return theta


def plackett_tau_to_theta(spearman):

    def bisection(ps):
        def funct(theta, ps):
            # Function to solve numerically.
            f = (ps - (theta + 1)/(theta - 1) + (2*theta*np.log(theta))/(theta - 1)**2)
            return f

        if -1 <= ps < 0:
            if -1 + 1e-8 < ps < 0:
                solution = sop.bisect(funct, 0+1e-10, 1-1e-10, args=(ps))
            else:
                solution = 0
        elif 0 < ps <= 1:
            if ps <= 1 - 1e-8:
                solution = sop.bisect(funct, 1+1e-10, 1e10, args=(ps))
            else:
                solution = np.inf
        else:
            solution = 1
        return solution

    ps = np.round(spearman, 3)
    theta = bisection(ps)

    return theta


def transformation_u_to_omega(u_vector, theta_plackett):

    standard_normal = st.norm(loc=0, scale=1)
    ind_vector = standard_normal.cdf(u_vector)
    ind_vector_c_phi = ind_vector[:, :2]
    # ind_vector_c_phi = ind_vector[:, :2]
    dep_vector = np.zeros((u_vector.shape[0], u_vector.shape[1]))

    dep_vector[:, 0], dep_vector[:, 1] = rvs_plackett(ind_vector_c_phi,
                                                      theta_plackett)
    dep_vector[:, 2] = ind_vector[:, 2]

    return dep_vector
