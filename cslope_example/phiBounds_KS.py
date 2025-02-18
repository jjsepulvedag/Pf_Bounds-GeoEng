r""" Example of Kolgomorov-Smirnov bounds

This script conducts an example of the Kolmogorov-Smirnov bounds applied to 
a soil friction angle dataset.

-------------------------------------------------------------------------------
Created by:  Juan Jose Sepulveda Garcia, Diego Andres Alvarez Marin
Mails:       jjsepulvedag@unal.edu.co,   daalvarez@unal.edu.co
Date:        January 2021
Institution: Universidad Nacional de Colombia (https://unal.edu.co/)
-------------------------------------------------------------------------------

Functions:
----------
dvalues_kstest :
empirical_cdf  :
upper_cdf      :
lower_cdf      :
focal_elements :

Notes:
----------

References:
----------
1."Infinite random sets and applications in uncertainty analysis"
Álvares M, Diego A.
2007
PhD. Thesis.
Leopold-Franzens Universitat Innsbruck.
Innsbruck, Austria.
"""

import numpy as np
import matplotlib.pyplot as plt


# VARIABLES DEFINED BY USER: 
# confidence level of the K-S bounds, it must be 0.8, 0.9, 0.95 or 0.99
# phi_min, phi_max: min and max values of phi, respectively
confidence_level = 0.80
phi_min = 7
phi_max = 37

# Data for the construction of the empirical CDF
phi = np.array([22, 23.2, 23.4, 24, 24, 24, 24.1, 24.3, 24.4, 24.9, 25, 25.3,
                25.5, 25.6, 26, 26.5, 27, 28.5, 29.5, 30])
# phi = np.array([12, 13.2, 13.4, 14, 14, 14, 14.1, 14.3, 14.4, 14.9, 15, 15.3,
#                 15.5, 15.6, 16, 16.5, 17, 18.5, 19.5, 20])



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
    dataset of phi taken from Oberguggenberger

    return the kolmogorov bounds according to a confidence level and a min and 
    max value of \phi
    '''
    focal_elements = np.zeros((len(alpha), 2))
    xmin_indexes = np.searchsorted(y_sup, alpha, side='left')
    xmax_indexes = np.searchsorted(y_inf, alpha, side='left')

    focal_elements[:,0] = phi_emp[xmin_indexes]
    focal_elements[:,1] = phi_emp[xmax_indexes]

    return focal_elements


    






