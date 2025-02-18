'''
This function takes an alpha vector (between 0 and 1) and return the 
corresponding focal elements according to a structure of uncertainty
'''
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def p_FocElems(alpha): 
    '''
    Structure of uncertainty: triangular PDF
    '''

    dist_r = st.triang(c=1/3, loc=800, scale=1500)
    focal_elements = dist_r.ppf(alpha)
    focal_elements = np.reshape(focal_elements, (len(focal_elements), 1))

    return focal_elements

# x = np.random.rand(100)
# y = p_FocElems(x)
# # print(x, y)

# plt.scatter(y, x)
# # plt.scatter(y[:,1], x)
# plt.show()
