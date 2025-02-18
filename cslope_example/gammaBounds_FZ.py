'''
Trapezoidal fuzzy set (16, 18, 20, 22)

therefore

alpha = (1/2)x_min - 8
alpha = -(1/2)x_max + 11

'''

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def gamma_FocElems(alpha): 
    '''
    This function takes an alpha [0,1] and returns a focal element delimited 
    by its lower and upper bounds. 
    '''

    # return np.transpose(np.array([2*(alpha+8), 2*(11-alpha)]))
    # return np.transpose(np.array([1.5*(alpha+2), 7*(2-alpha)]))
    # return np.transpose(np.array([1/20*(alpha+2), 1/20*(5-alpha)]))
    x = st.norm.ppf(alpha, loc=18, scale=1)
    y = st.norm.ppf(alpha, loc=22, scale=2)

    return np.transpose(np.array([x, y]))



# x = np.random.rand(1000)
# y = gamma_FocElems(x)


# plt.scatter(y[:,0],x)
# plt.scatter(y[:,1],x)
# plt.show()