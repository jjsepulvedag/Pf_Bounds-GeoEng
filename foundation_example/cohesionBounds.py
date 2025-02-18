'''
This function takes an alpha vector (between 0 and 1) and return the 
corresponding focal elements according to a structure of uncertainty


Data set from Li et al (2015): Bivariate distribution of shear strength 
parameters using copulas and its impact on geotechnical system reliability

'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def c_FocElems(alpha): 
    '''
    This function takes an alpha [0,1] and returns a focal element delimited 
    by its lower and upper bounds. 
    '''

    x = st.uniform.ppf(alpha, loc=10, scale=10)
    y = st.uniform.ppf(alpha, loc=15, scale=10)


    return np.transpose(np.array([x, y]))

# x = np.random.rand(1000)
# y = c_FocElems(x)
# # print(x, y)

# plt.scatter(y[:,0], x)
# plt.scatter(y[:,1], x)
# plt.show()
