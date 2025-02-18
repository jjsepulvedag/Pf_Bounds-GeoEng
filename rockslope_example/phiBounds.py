'''
This function takes an alpha vector (between 0 and 1) and return the 
corresponding focal elements according to a structure of uncertainty


Data set from Li et al (2015): Bivariate distribution of shear strength 
parameters using copulas and its impact on geotechnical system reliability

'''
import numpy as np
import matplotlib.pyplot as plt


# VARIABLES DEFINED BY USER: 
# confidence level of the K-S bounds, it must be 0.8, 0.9, 0.95 or 0.99
# phi_min, phi_max: min and max values of phi, respectively
confidence_level = 0.80
phi_min = 15
phi_max = 35

# Data for the construction of the empirical CDF
phi = np.array([20.78, 25.45, 22.22, 23.30, 24.78, 20.06, 18.77, 24.76, 18.49, 
                22.95, 24.75, 23.66, 17.02, 23.16, 20.07, 16.63, 21.03, 23.62, 
                22.70, 20.43, 21.41, 22.81, 22.14, 22.28, 22.13, 22.87, 24.00, 
                18.70, 22.33, 21.37, 21.33, 22.75, 19.92, 19.20, 22.68, 21.56, 
                26.14, 23.44, 22.05, 22.47, 20.58, 22.50, 20.65, 27.34, 21.14, 
                23.63, 22.83, 20.28, 21.19, 23.09, 22.56, 23.77, 27.03, 25.60, 
                23.74, 21.44, 27.31, 26.81, 26.18, 23.29, 22.62, 30.32, 27.20])
                #units: °



# Compute variables (not defined by the user) ----------------------

alpha_KS = np.round(1.0 - confidence_level, 2)  # alpha_KS from the K-S test

phi_emp = np.sort(phi)  # Sorting the dataset
n = phi_emp.shape[0]  # Number of samples in the dataset

if alpha_KS == 0.01:
    D_ks =  1.63/np.sqrt(n)
elif alpha_KS == 0.05:
    D_ks = 1.36/np.sqrt(n)
elif alpha_KS == 0.10:
    D_ks = 1.22/np.sqrt(n)
elif alpha_KS == 0.20:
    D_ks = 1.07/np.sqrt(n)
else: 
    print('Invalid confidence level')

y_emp = np.arange(0, 1, 1/n) + 1/n

y_sup = [i+D_ks if i + D_ks <= 1 else 1 for i in y_emp]
y_sup = np.insert(y_sup, [0, len(y_sup)], [D_ks, y_sup[-1]])
y_inf = [i-D_ks if i - D_ks >= 0 else 0 for i in y_emp]
y_inf = np.insert(y_inf, [0, len(y_inf)], [y_inf[0], 1])

phi_emp = np.insert(phi_emp, [0, len(phi_emp)], [phi_min, phi_max])



def phi_FocElems(alpha):
    '''
    dataset of phi/cohesion taken from Li et al (2015)

    Return the focal elements according to a vector alpha and the K-S bounds
    previously defined.
    '''
    focal_elements = np.zeros((len(alpha), 2))
    xmin_indexes = np.searchsorted(y_sup, alpha, side='left')
    xmax_indexes = np.searchsorted(y_inf, alpha, side='left')

    focal_elements[:,0] = phi_emp[xmin_indexes]
    focal_elements[:,1] = phi_emp[xmax_indexes]

    return focal_elements

# # print(min(phi))
# # print(max(phi))
# y = np.random.rand(1000)
# x = phi_FocElems(y)

# plt.scatter(x[:,0], y)
# plt.scatter(x[:,1], y)
# plt.show()