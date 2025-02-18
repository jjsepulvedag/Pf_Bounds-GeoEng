'''
This function takes an alpha vector (between 0 and 1) and return the 
corresponding focal elements according to a structure of uncertainty
'''
import numpy as np
# import matplotlib.pyplot as plt

def g_FocElems(alpha): 
    '''
    Structure of uncertainty: Trapezoidal Fuzzy set [23 25 27 29] kN/m3
    '''

    return np.transpose(np.array([2*(alpha+11.5), 2*(14.5-alpha)]))

# x = np.random.rand(1000)
# y = gamma_FocElems(x)
# print(y)

# plt.scatter(y[:,0], x)
# plt.scatter(y[:,1], x)
# plt.show()
