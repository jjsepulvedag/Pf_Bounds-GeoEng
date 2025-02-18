'''
This function takes an alpha vector (between 0 and 1) and return the 
corresponding focal elements according to a structure of uncertainty


Data set from Li et al (2015): Bivariate distribution of shear strength 
parameters using copulas and its impact on geotechnical system reliability

'''
import numpy as np
import matplotlib.pyplot as plt


def phi_FocElems(alpha): 
    '''
    Structure of uncertainty: Family of equally credible intervals
    '''

    z_intervals = np.array([[10, 12], [11, 13], [12, 13]])
    z_alphas = np.array([0.333, 0.666, 1])

    focal_elements = np.zeros((len(alpha), 2))
    
    alpha_index = np.searchsorted(z_alphas, alpha, side='left')

    focal_elements[:,:] = z_intervals[alpha_index]

    return focal_elements

# y = np.random.rand(1000)
# x = phi_FocElems(y)

# plt.scatter(x[:,0], y)
# plt.scatter(x[:,1], y)
# plt.show()