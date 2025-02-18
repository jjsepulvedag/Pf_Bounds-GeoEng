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
c_min = 10
c_max = 115

# Data for the construction of the empirical CDF
cohesion = np.array([75.41, 45.72, 57.40, 81.43, 67.32, 56.15, 92.19, 65.97, 
                     64.56, 20.71, 27.43, 80.83, 49.37, 36.39, 75.74, 81.04, 
                     71.61, 69.81, 53.50, 88.02, 93.48, 74.30, 65.03, 62.27, 
                     56.54, 77.68, 71.40, 86.09, 58.35, 91.49, 91.52, 18.91, 
                     81.19, 64.93, 59.37, 105.58, 56.38, 51.75, 87.80, 83.94, 
                     112.53, 61.65, 85.44, 38.82, 70.72, 91.14, 63.85, 87.13, 
                     79.26, 46.37, 80.48, 81.03, 33.87, 51.75, 71.57, 101.74, 
                     30.67, 48.77, 52.46, 43.74, 72.41, 37.68, 14.57]) #kPa



# Compute variables (not defined by the user) ----------------------

alpha_KS = np.round(1.0 - confidence_level, 2)  # alpha_KS from the K-S test

c_emp = np.sort(cohesion)  # Sorting the dataset
n = c_emp.shape[0]  # Number of samples in the dataset

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

c_emp = np.insert(c_emp, [0, len(c_emp)], [c_min, c_max])



def c_FocElems(alpha):
    '''
    dataset of cohesion taken from Li et al (2015)

    Return the focal elements according to a vector alpha and the K-S bounds
    previously defined.
    '''
    focal_elements = np.zeros((len(alpha), 2))
    xmin_indexes = np.searchsorted(y_sup, alpha, side='left')
    xmax_indexes = np.searchsorted(y_inf, alpha, side='left')

    focal_elements[:,0] = c_emp[xmin_indexes]
    focal_elements[:,1] = c_emp[xmax_indexes]

    return focal_elements

# # print(min(cohesion))
# # print(max(cohesion))
# y = np.random.rand(1000)
# x = c_FocElems(y)

# plt.scatter(x[:,0], y)
# plt.scatter(x[:,1], y)
# plt.show()