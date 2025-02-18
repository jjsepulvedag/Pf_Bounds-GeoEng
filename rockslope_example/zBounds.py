'''
This function takes an alpha vector (between 0 and 1) and return the 
corresponding focal elements according to a structure of uncertainty
'''
import numpy as np
import matplotlib.pyplot as plt

def z_FocElems(alpha): 
    '''
    Structure of uncertainty: Family of equally credible intervals
    '''

    z_intervals = np.array([[1, 6], [5, 8], [7, 10]])
    z_alphas = np.array([0.333, 0.666, 1])

    focal_elements = np.zeros((len(alpha), 2))
    
    alpha_index = np.searchsorted(z_alphas, alpha, side='left')

    focal_elements[:,:] = z_intervals[alpha_index]

    return focal_elements

# x = np.random.rand(100)
# y = gamma_FocElems(x)
# # print(x, y)

# plt.scatter(y[:,0], x)
# plt.scatter(y[:,1], x)
# plt.show()
