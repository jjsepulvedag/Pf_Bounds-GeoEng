'''
This function takes an alpha vector (between 0 and 1) and return the 
corresponding focal elements according to a structure of uncertainty
'''
import numpy as np
import scipy.stats as st
# import matplotlib.pyplot as plt

def kh_FocElems(alpha): 
    '''
    Structure of uncertainty: triangular PDF
    '''

    dist_kh = st.uniform(loc=0.05, scale=0.10)
    focal_elements = dist_kh.ppf(alpha)
    focal_elements = np.reshape(focal_elements, (len(focal_elements), 1))

    return focal_elements

# x = np.random.rand(100)
# y = gamma_FocElems(x)
# # print(x, y)

# plt.scatter(y[:,0], x)
# plt.scatter(y[:,1], x)
# plt.show()
